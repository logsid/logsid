"""Native Unicode span SLM for SCOPE, with resumable few-shot fine tuning.

The encoder uses a pinned RoBERTa backbone, with reproducible RNG state and
atomic checkpoint helpers from checkpoint.py. The output representation uses
character contextual states + a GlobalPointer endpoint scorer, never BIO/MLM
projection. Saved metadata records the backbone revision and source commit.
"""
from __future__ import annotations

import json
import math
import time
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_scheduler, set_seed

from .io import DEFAULT_CACHE, atomic_json, span_set
from .checkpoint import (COMMIT, MODEL_ID, MODEL_REVISION, _atomic_torch_save,
                              _restore_rng, _rng_state)


DEFAULT_MODEL = (DEFAULT_CACHE / "huggingface" / "models--FacebookAI--roberta-base"
                 / "snapshots" / MODEL_REVISION)


def _rope(x):
    """Rotary endpoint positions; final dimension must be even."""
    width = x.shape[-1]
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    scales = torch.exp(-math.log(10000) * torch.arange(0, width, 2,
                       device=x.device, dtype=torch.float32) / width)
    angles = positions[:, None] * scales[None, :]
    sin, cos = angles.sin().to(x.dtype), angles.cos().to(x.dtype)
    left, right = x[..., 0::2], x[..., 1::2]
    return torch.stack((left * cos - right * sin,
                        right * cos + left * sin), dim=-1).flatten(-2)


def multilabel_span_loss(logits, labels, mask):
    """GP multilabel logsumexp loss; gold keeps all nested/overlapping spans.

    Each category independently discriminates all valid character endpoint
    pairs against zero. There is no sampled-negative/BIO label projection.
    """
    values = logits.float().flatten(-2)
    targets = labels.flatten(-2).bool()
    valid = mask[:, None].flatten(-2).expand_as(targets)
    zero = torch.zeros_like(values[..., :1])
    negative = torch.cat((values.masked_fill(~(valid & ~targets), -torch.inf), zero), -1)
    positive = torch.cat((-values.masked_fill(~(valid & targets), torch.inf), zero), -1)
    return (torch.logsumexp(negative, -1) + torch.logsumexp(positive, -1)).mean()


class CharacterPointer(nn.Module):
    """Every Python Unicode codepoint has a start and inclusive-end endpoint."""
    def __init__(self, encoder, categories, config):
        super().__init__()
        self.encoder = encoder
        self.categories = categories
        self.head_dim = int(config["head_dim"])
        self.char_vocab = int(config["char_vocab"])
        char_dim, hidden = int(config["char_dim"]), int(config["hidden_dim"])
        self.char_embedding = nn.Embedding(self.char_vocab + 1, char_dim, padding_idx=0)
        self.char_cnn = nn.Conv1d(char_dim, char_dim, 3, padding=1)
        self.source_embedding = (nn.Embedding(len(config["source_names"]), config["source_dim"])
                                 if config["source_names"] else None)
        source_dim = config["source_dim"] if self.source_embedding is not None else 0
        self.combine = nn.Sequential(nn.Linear(encoder.config.hidden_size + char_dim + source_dim,
                                              hidden), nn.GELU(), nn.Dropout(config["dropout"]))
        self.endpoints = nn.Linear(hidden, categories * self.head_dim * 2)
        self.presence = nn.Linear(encoder.config.hidden_size + source_dim, categories)
        self.freeze_encoder = not config["train_encoder"]
        if self.freeze_encoder:
            encoder.requires_grad_(False)
        else:
            layers = config.get("encoder_layers_to_train")
            if layers is not None:
                # RoBERTa family, explicitly reject other architectures.
                if not hasattr(encoder, "encoder") or not hasattr(encoder.encoder, "layer"):
                    raise ValueError("encoder_layers_to_train requires a RoBERTa encoder")
                count = len(encoder.encoder.layer)
                if not 0 <= layers <= count:
                    raise ValueError("Invalid trainable encoder layer count")
                encoder.requires_grad_(False)
                for layer in list(encoder.encoder.layer)[count - layers:]:
                    if layers:
                        layer.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, batch):
        hidden = self.encoder(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"]).last_hidden_state
        gather = batch["char_tokens"][..., None].expand(-1, -1, hidden.shape[-1])
        context = hidden.gather(1, gather)
        characters = self.char_embedding(batch["char_ids"])
        characters = torch.nn.functional.gelu(self.char_cnn(characters.transpose(1, 2))).transpose(1, 2)
        components = [context, characters]
        source = None
        if self.source_embedding is not None:
            source = self.source_embedding(batch["source_ids"])
            components.append(source[:, None, :].expand(-1, context.shape[1], -1))
        features = self.combine(torch.cat(components, -1))
        endpoints = self.endpoints(features).reshape(*features.shape[:2],
                                  self.categories, 2, self.head_dim)
        query, key = endpoints[:, :, :, 0].permute(0, 2, 1, 3), endpoints[:, :, :, 1].permute(0, 2, 1, 3)
        query, key = _rope(query), _rope(key)
        logits = torch.einsum("bcid,bcjd->bcij", query, key) / math.sqrt(self.head_dim)
        # Padding and end < start cannot be decoded or enter the loss.
        lengths = batch["char_lengths"]
        index = torch.arange(logits.shape[-1], device=logits.device)
        valid = ((index[None, :, None] <= index[None, None, :])
                 & (index[None, :, None] < lengths[:, None, None])
                 & (index[None, None, :] < lengths[:, None, None]))
        pooled = (hidden * batch["attention_mask"][..., None]).sum(1)
        pooled = pooled / batch["attention_mask"].sum(1).clamp_min(1)[:, None]
        if source is not None:
            pooled = torch.cat((pooled, source), -1)
        return logits.masked_fill(~valid[:, None], -torch.inf), valid, self.presence(pooled)


class ScopeSLM:
    """RoBERTa/character GlobalPointer; score is uncalibrated set ranking only.

    fit uses ONLY supplied train/validation records. The caller must choose both
    from the historical training20, keeping chronological test labels private.
    """
    DEFAULTS = {
        "model_path": str(DEFAULT_MODEL), "device": "cpu", "seed": 42,
        "max_chars": 512, "max_tokens": 256, "head_dim": 32, "hidden_dim": 192,
        "char_dim": 48, "char_vocab": 4096, "dropout": 0.1,
        "epochs": 3, "batch_size": 4, "inference_batch_size": 8,
        "learning_rate": 3e-5, "head_learning_rate": 3e-4, "weight_decay": 0.01,
        "warmup_ratio": 0.1, "max_grad_norm": 1.0, "checkpoint_every": 100,
        "progress_seconds": 30, "mixed_precision": "no", "gradient_checkpointing": False,
        "train_encoder": True, "encoder_layers_to_train": None,
        "presence_weight": 0.1, "threshold": 0.0,
        "pair_loss_weight": 0.0,  # Retained for saved configs and resume identities.
        "max_predictions": 256, "stop_after_steps": None,
        "local_files_only": True, "source_names": [], "source_dim": 16,
    }

    def __init__(self, categories, config=None):
        self.categories = list(categories)
        if not self.categories or len(self.categories) != len(set(self.categories)):
            raise ValueError("SLM requires unique ordered category names")
        supplied = dict(config or {})
        if set(supplied) - set(self.DEFAULTS):
            raise ValueError("Unsupported SLM config: " + str(sorted(set(supplied) - set(self.DEFAULTS))))
        self.config = {**self.DEFAULTS, **supplied}
        cfg = self.config
        cfg["source_names"] = list(cfg["source_names"])
        if (len(cfg["source_names"]) != len(set(cfg["source_names"]))
                or any(not isinstance(name, str) or not name for name in cfg["source_names"])):
            raise ValueError("source_names must be distinct nonempty strings")
        cfg["model_path"], cfg["device"] = str(cfg["model_path"]), str(cfg["device"])
        if cfg["pair_loss_weight"] != 0:
            raise ValueError("Natural aligned-field pair loss is not implemented; pair_loss_weight must be 0")
        for key in ("max_chars", "max_tokens", "head_dim", "hidden_dim", "char_dim", "char_vocab",
                    "epochs", "batch_size", "inference_batch_size", "checkpoint_every", "max_predictions", "source_dim"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(key + " must be a positive integer")
        if cfg["head_dim"] % 2:
            raise ValueError("head_dim must be even for RoPE")
        if cfg["mixed_precision"] not in ("no", "bf16", "fp16"):
            raise ValueError("mixed_precision must be no/bf16/fp16")
        self.device = torch.device(cfg["device"])
        if self.device.type != "cuda" and cfg["mixed_precision"] != "no":
            raise ValueError("Mixed precision currently requires CUDA")
        set_seed(cfg["seed"])
        model_path = str(Path(cfg["model_path"]).expanduser())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                              local_files_only=cfg["local_files_only"])
        if not self.tokenizer.is_fast:
            raise ValueError("Unicode alignment requires a fast tokenizer")
        if getattr(self, "_portable_config", None) is not None:
            encoder = AutoModel.from_config(self._portable_config, add_pooling_layer=False)
        else:
            encoder = AutoModel.from_pretrained(model_path, local_files_only=cfg["local_files_only"],
                                                add_pooling_layer=False)
        if cfg["max_tokens"] > encoder.config.max_position_embeddings - 2:
            raise ValueError("max_tokens exceeds RoBERTa positional capacity")
        if cfg["gradient_checkpointing"]:
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = CharacterPointer(encoder, len(self.categories), cfg).to(self.device)
        self.model.eval()
        self.trained = False
        self.prediction_rows = 0
        self.training = {}
        self.metadata = {
            "method": "SCOPE-character-GlobalPointer", "backbone": MODEL_ID,
            "backbone_revision": MODEL_REVISION, "unleash_source_commit": COMMIT,
            "representation": "All Unicode character endpoint pairs, independent categories; supports nesting and overlaps",
            "char_embedding": "ord(codepoint) modulo char_vocab plus 1; CNN contextualized, hash collisions possible",
            "long_input": "Overlapping complete-coverage windows; original coordinates and minimum window score",
            "score": "Uncalibrated ranking of complete-output uncertainty, including empty predictions; not correctness probability",
            "risk_head": "No OOF exact-set/omission risk head in this module; raw features require caller calibration",
            "source_conditioning": {"names": cfg["source_names"], "dimensions": cfg["source_dim"],
                                     "injection": "Learned source embedding concatenated to character and presence features; raw text offsets unchanged"},
            "config": dict(cfg), "trained": False,
        }

    def _source_id(self, source):
        names = self.config["source_names"]
        if not names:
            return 0
        if source is None and len(names) == 1:
            return 0
        if source is None:
            raise ValueError("SLM source is required for a multi-source checkpoint")
        if source not in names:
            raise ValueError("Unknown SLM source; checkpoint source vocabulary is frozen")
        return names.index(source)

    def _encode(self, text, source=None):
        if not isinstance(text, str):
            raise TypeError("SLM input must be raw Content str")
        source_id = self._source_id(source)
        if len(text) > self.config["max_chars"]:
            return None, "character-window"
        encoded = self.tokenizer(text, truncation=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) > self.config["max_tokens"]:
            return None, "token-window"
        offsets = encoded.pop("offset_mapping")
        mapping = [-1] * len(text)
        for token, (start, end) in enumerate(offsets):
            for char in range(start, min(end, len(text))):
                if mapping[char] == -1:
                    mapping[char] = token
        # Whitespace/control characters sometimes lack a BPE offset; they keep
        # their own character embedding and receive neighboring token context.
        previous = 0
        for index in range(len(mapping)):
            if mapping[index] >= 0:
                previous = mapping[index]
            else:
                mapping[index] = previous
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
                "char_tokens": mapping, "char_ids": [ord(c) % self.config["char_vocab"] + 1 for c in text],
                "char_length": len(text), "text": text, "source_id": source_id}, None

    def _collate(self, encoded, labels=None):
        tokens = max(len(row["input_ids"]) for row in encoded)
        # A zero-character log is supported, with no valid span lattice cells.
        chars = max(1, max(row["char_length"] for row in encoded))
        batch = {}
        for key, pad, length in (("input_ids", self.tokenizer.pad_token_id, tokens),
                                 ("attention_mask", 0, tokens), ("char_tokens", 0, chars),
                                 ("char_ids", 0, chars)):
            batch[key] = torch.tensor([row[key] + [pad] * (length - len(row[key]))
                                       for row in encoded], dtype=torch.long, device=self.device)
        batch["char_lengths"] = torch.tensor([row["char_length"] for row in encoded],
                                             device=self.device)
        batch["source_ids"] = torch.tensor([row["source_id"] for row in encoded],
                                           device=self.device)
        if labels is not None:
            targets = torch.zeros((len(encoded), len(self.categories), chars, chars),
                                   dtype=torch.bool, device=self.device)
            for row, spans in enumerate(labels):
                for start, end, category in span_set(encoded[row]["text"], spans, self.categories):
                    targets[row, self.categories.index(category), start, end - 1] = True
            batch["labels"] = targets
        return batch

    def _autocast(self):
        dtype = torch.bfloat16 if self.config["mixed_precision"] == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype,
                              enabled=self.config["mixed_precision"] != "no")

    def _loss(self, batch):
        with self._autocast():
            logits, valid, presence = self.model(batch)
            span_loss = multilabel_span_loss(logits, batch["labels"], valid)
            observed = batch["labels"].flatten(-2).any(-1).float()
            auxiliary = torch.nn.functional.binary_cross_entropy_with_logits(presence.float(), observed)
            return span_loss + self.config["presence_weight"] * auxiliary

    def _windows(self, text, source):
        offsets = self.tokenizer(text, add_special_tokens=False,
                                 return_offsets_mapping=True)['offset_mapping']
        ends, windows, first = [end for _, end in offsets], [], 0
        while first < len(offsets):
            start = 0 if first == 0 else offsets[first][0]
            last = min(len(offsets), first + self.config['max_tokens'] - 2,
                       bisect_right(ends, start + self.config['max_chars']))
            if last <= first:
                return None
            end = len(text) if last == len(offsets) else ends[last - 1]
            # Re-tokenization at a new boundary can add tokens.
            while self._encode(text[start:end], source)[1]:
                last -= 1
                if last <= first:
                    return None
                end = ends[last - 1]
            windows.append((start, end))
            if end == len(text):
                break
            first += max(1, (last - first) // 2)
        if (not windows or windows[0][0] != 0 or windows[-1][1] != len(text)
                or any(a[1] < b[0] for a, b in zip(windows, windows[1:]))):
            return None
        return windows

    def predict_batch(self, texts, sources=None):
        self.prediction_rows += len(texts)
        output = self._predict_batch(texts, sources)
        for index, (text, prediction) in enumerate(zip(texts, output)):
            if prediction['risk_features'].get('reason') not in ('character-window', 'token-window'):
                continue
            source = sources[index] if sources is not None else None
            windows = self._windows(text, source)
            if windows is None:
                continue
            parts = self._predict_batch([text[a:b] for a, b in windows], [source] * len(windows))
            if any(part['spans'] is None for part in parts):
                continue
            spans = set()
            for i, ((start, end), part) in enumerate(zip(windows, parts)):
                left = (start + windows[i - 1][1]) / 2 if i else 0
                right = (end + windows[i + 1][0]) / 2 if i + 1 < len(windows) else len(text) + 1
                # Only the window with the most context around a midpoint emits
                # that span; all windows still contribute their uncertainty.
                for span in part['spans']:
                    first, last = start + span['start'], start + span['end']
                    if left <= (first + last) / 2 < right:
                        spans.add((first, last, span['category']))
            if len(spans) > self.config['max_predictions']:
                output[index] = {'spans': None, 'score': 0.0, 'route': 'slm-unsupported',
                                 'risk_features': {'supported': False, 'reason': 'prediction-overflow'}}
                continue
            output[index] = {'spans': [{'start': a, 'end': b, 'category': c}
                                      for a, b, c in sorted(spans)],
                             'score': min(part['score'] for part in parts), 'route': 'slm',
                             'risk_features': {'supported': True, 'characters': len(text),
                                               'windows': len(windows), 'score_is_calibrated': False}}
        return output

    def _predict_batch(self, texts, sources=None):
        if not self.trained:
            raise ValueError("SLM has no completed trained checkpoint; refusing random predictions")
        if sources is None:
            if len(self.config["source_names"]) > 1:
                raise ValueError("SLM sources are required for a multi-source checkpoint")
            sources = [None] * len(texts)
        if len(sources) != len(texts):
            raise ValueError("SLM sources and texts must have equal length")
        output, pending = [None] * len(texts), []
        for index, (text, source) in enumerate(zip(texts, sources)):
            try:
                self._source_id(source)
            except ValueError:
                output[index] = {"spans": None, "score": 0.0, "route": "slm-unsupported",
                                 "risk_features": {"supported": False, "reason": "unknown-source"}}
                continue
            encoded, reason = self._encode(text, source)
            if reason:
                output[index] = {"spans": None, "score": 0.0, "route": "slm-unsupported",
                                 "risk_features": {"supported": False, "reason": reason,
                                                   "characters": len(text)}}
            else:
                pending.append((index, encoded))
        self.model.eval()
        batch_size = self.config["inference_batch_size"]
        with torch.inference_mode():
            for offset in range(0, len(pending), batch_size):
                rows = pending[offset:offset + batch_size]
                batch = self._collate([row[1] for row in rows])
                with self._autocast():
                    logits, valid, presence_logits = self.model(batch)
                # Reduce the full endpoint lattice where it was computed.
                # Transferring B*C*L*L floating scores to the CPU made inference
                # memory-bandwidth bound; only compact summaries and selected
                # coordinates need to cross the device boundary.
                scores = logits.float()
                selected = scores > self.config["threshold"]
                counts = selected.sum(dim=(1, 2, 3))
                summaries = torch.stack(((scores - self.config["threshold"]).abs().amin(dim=(1, 2, 3)),
                    scores.amax(dim=(1, 2, 3)),
                    scores.masked_fill(selected, -torch.inf).amax(dim=(1, 2, 3))), dim=-1).cpu().tolist()
                category_flags = selected.flatten(-2).any(-1).cpu().tolist()
                allowed = counts <= self.config["max_predictions"]
                coordinates = (selected & allowed[:, None, None, None]).nonzero().cpu().tolist()
                found_by_row = [[] for _ in rows]
                for row_index, category, start, end in coordinates:
                    found_by_row[row_index].append([category, start, end])
                counts = counts.cpu().tolist()
                presence_all = presence_logits.float().sigmoid().cpu()
                for row, (index, encoded) in enumerate(rows):
                    presence = presence_all[row]
                    count = counts[row]
                    if count > self.config["max_predictions"]:
                        # Count before allocating a potentially enormous Python
                        # endpoint list when an underfit head emits dense spans.
                        categories = [i for i, found in enumerate(category_flags[row]) if found]
                        features = self._risk_features_from_summary(*summaries[row], presence,
                            [(category, 0, 0) for category in categories], encoded["char_length"], span_count=count)
                        output[index] = {"spans": None, "score": 0.0, "route": "slm-unsupported",
                                         "risk_features": {**features, "supported": False,
                                                           "reason": "prediction-overflow"}}
                        continue
                    found = found_by_row[row]
                    features = self._risk_features_from_summary(*summaries[row], presence, found, encoded["char_length"])
                    spans = [{"start": start, "end": end + 1, "category": self.categories[category]}
                             for category, start, end in found]
                    spans.sort(key=lambda span: (span["start"], span["end"], span["category"]))
                    output[index] = {"spans": spans, "score": features["set_ranking_score"],
                                     "risk_features": features, "route": "slm"}
        return output

    def _risk_features_from_summary(self, margin, maximum_logit, excluded_logit,
                                    presence, found, characters, span_count=None):
        if maximum_logit == -math.inf:  # Empty Content has no valid endpoints.
            margin, maximum, omission, confidence = 0.0, 0.0, 0.0, 0.0
        else:
            maximum = float(torch.tensor(maximum_logit).sigmoid())
            omission = float(torch.tensor(excluded_logit).sigmoid()) if excluded_logit != -math.inf else 0.0
            confidence = float(torch.sigmoid(torch.tensor(margin)) * 2 - 1)
        predicted_categories = {item[0] for item in found}
        disagreement = max(float(abs(presence[i] - float(i in predicted_categories)))
                           for i in range(len(self.categories)))
        # Auxiliary category-presence disagreement can lower a confidently empty
        # lattice; it is neither an OOF error model nor a calibrated probability.
        ranking = confidence * (1.0 - disagreement)
        return {"supported": True, "characters": characters,
                "span_count": len(found) if span_count is None else span_count,
                "empty": not found, "minimum_endpoint_margin": margin,
                "maximum_span_probability": maximum, "maximum_excluded_probability": omission,
                "presence_maximum": float(presence.max()), "presence_disagreement": disagreement,
                "presence_by_category": {name: float(presence[i]) for i, name in enumerate(self.categories)},
                "set_ranking_score": ranking, "score_is_calibrated": False}

    def fit(self, train_records, validation_records, output_dir, resume=True):
        """Resumable exact-batch training; validation alone selects epoch weights.

        stop_after_steps is an optional cumulative optimizer-step test/debug
        limit. An interrupted run returns status=incomplete, cannot predict,
        and resumes with identical data/config from latest.pt.
        """
        fit_started = time.perf_counter()
        train_records, validation_records = list(train_records), list(validation_records)
        if not train_records or not validation_records:
            raise ValueError("Nonempty historical fit and independent validation records required")
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        encoded_train, encoded_validation, skipped = [], [], Counter()
        category_counts = Counter()
        for role, rows, target in (("train", train_records, encoded_train),
                                    ("validation", validation_records, encoded_validation)):
            for record in rows:
                # Invalid or unrepresentable gold is never projected away.
                spans = span_set(record["text"], record["spans"], self.categories)
                encoded, reason = self._encode(record["text"], record.get("source"))
                if reason:
                    skipped[role + "_" + reason] += 1
                    continue
                target.append((encoded, record["spans"]))
                if role == "train":
                    category_counts.update(category for _, _, category in spans)
        if not encoded_train or not encoded_validation:
            raise ValueError("No supported historical fit/validation records in SLM windows")
        batches = math.ceil(len(encoded_train) / self.config["batch_size"])
        total_steps = batches * self.config["epochs"]
        ignored = {"device", "inference_batch_size", "checkpoint_every", "progress_seconds", "stop_after_steps"}
        identity = {"categories": self.categories, "config": {key: value for key, value in self.config.items() if key not in ignored},
                    "train_rows": len(train_records), "validation_rows": len(validation_records),
                    "sampling": self.metadata.get("sampling")}
        parameters = list(self.model.named_parameters())
        optimizer = torch.optim.AdamW([
            {"params": [p for name, p in parameters if p.requires_grad and name.startswith("encoder.")],
             "lr": self.config["learning_rate"], "weight_decay": self.config["weight_decay"]},
            {"params": [p for name, p in parameters if p.requires_grad and not name.startswith("encoder.")],
             "lr": self.config["head_learning_rate"], "weight_decay": self.config["weight_decay"]},
        ], betas=(0.9, 0.999), eps=1e-8)
        scheduler = get_scheduler("linear", optimizer=optimizer,
                                  num_warmup_steps=int(total_steps * self.config["warmup_ratio"]),
                                  num_training_steps=total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.config["mixed_precision"] == "fp16")
        training = {"status": "running", "optimizer_steps_completed": 0, "optimizer_steps_expected": total_steps,
                    "epochs_completed": 0, "train_supported": len(encoded_train), "validation_supported": len(encoded_validation),
                    "train_supplied": len(train_records), "validation_supplied": len(validation_records),
                    "unsupported_records": dict(skipped), "span_counts_by_category": dict(category_counts),
                    "categories_without_positive_training_spans": [name for name in self.categories if not category_counts[name]],
                    "validation": [], "best_validation_loss": None, "best_epoch": None,
                    "rows_seen_by_epoch": [0] * self.config["epochs"],
                    "protocol": "Caller-supplied historical fit/calibration subsets; no evaluation labels accessed"}
        epoch, next_batch = 0, 0
        latest, best = directory / "latest.pt", directory / "best-state.pt"
        if latest.exists():
            if not resume:
                raise ValueError("Checkpoint exists; resume it or choose a new output directory")
            state = torch.load(latest, map_location="cpu", weights_only=False)
            if state["identity"] != identity:
                raise ValueError("SLM resume configuration differs; use a new output directory")
            self.model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            training, epoch, next_batch = state["training"], state["epoch"], state["next_batch"]
            _restore_rng(state["rng"])
            del state
        committed_seconds = float(training.get("seconds", 0.0))
        training.setdefault("seconds", committed_seconds)
        training.setdefault("seconds_scope", "Cumulative fit-stage wall time at committed checkpoints; "
                            "excludes uncommitted interrupted work and final export")
        training["status"] = "running"
        self.trained = False

        def checkpoint():
            seconds = committed_seconds + time.perf_counter() - fit_started
            _atomic_torch_save(latest, {"identity": identity, "model": self.model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": _rng_state(), "training": {**training, "seconds": seconds},
                "epoch": epoch, "next_batch": next_batch})
            training["seconds"] = seconds
            atomic_json(directory / "training.json", training)

        last_report = time.monotonic()
        try:
            while epoch < self.config["epochs"]:
                generator = torch.Generator().manual_seed(self.config["seed"] + epoch)
                order = torch.randperm(len(encoded_train), generator=generator).tolist()
                self.model.train()
                for batch_index in range(next_batch, batches):
                    indices = order[batch_index * self.config["batch_size"]:(batch_index + 1) * self.config["batch_size"]]
                    selected = [encoded_train[i] for i in indices]
                    batch = self._collate([row[0] for row in selected], [row[1] for row in selected])
                    optimizer.zero_grad(set_to_none=True)
                    loss = self._loss(batch)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite SCOPE span loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    training["optimizer_steps_completed"] += 1
                    training["rows_seen_by_epoch"][epoch] += len(selected)
                    training["last_training_loss"] = float(loss.detach())
                    next_batch = batch_index + 1
                    if training["optimizer_steps_completed"] % self.config["checkpoint_every"] == 0:
                        checkpoint()
                    if time.monotonic() - last_report > self.config["progress_seconds"]:
                        print(json.dumps({"operation": "scope-slm-train", "epoch": epoch + 1,
                            "steps": training["optimizer_steps_completed"], "total_steps": total_steps,
                            "loss": training["last_training_loss"]}), flush=True)
                        last_report = time.monotonic()
                    stop = self.config["stop_after_steps"]
                    if stop is not None and training["optimizer_steps_completed"] >= stop:
                        training["status"] = "incomplete"
                        checkpoint()
                        self.training = training
                        return training
                if training["rows_seen_by_epoch"][epoch] != len(encoded_train):
                    raise ValueError("SLM epoch cardinality/resume mismatch")
                self.model.eval()
                measured, seen = 0.0, 0
                with torch.inference_mode():
                    for offset in range(0, len(encoded_validation), self.config["batch_size"]):
                        selected = encoded_validation[offset:offset + self.config["batch_size"]]
                        batch = self._collate([row[0] for row in selected], [row[1] for row in selected])
                        loss = float(self._loss(batch))
                        measured += loss * len(selected)
                        seen += len(selected)
                measured /= seen
                training["validation"].append({"epoch": epoch + 1, "span_presence_loss": measured, "rows": seen})
                if training["best_validation_loss"] is None or measured < training["best_validation_loss"]:
                    _atomic_torch_save(best, self.model.state_dict())
                    training["best_validation_loss"], training["best_epoch"] = measured, epoch + 1
                epoch += 1
                next_batch = 0
                training["epochs_completed"] = epoch
                checkpoint()
            self.model.load_state_dict(torch.load(best, map_location=self.device, weights_only=True))
            self.trained = True
            training["status"] = "complete"
            self.training = training
            self.metadata.update(trained=True, training=training)
            self.save(directory)
            atomic_json(directory / "training.json", training)
        except BaseException:
            # Latest checkpoint is a completed batch boundary. Never save a model
            # partially updated by an exception as if that boundary committed.
            training["status"] = "interrupted"
            atomic_json(directory / "training.json", training)
            raise
        finally:
            self.model.eval()
        return training

    def save(self, output_dir):
        if not self.trained:
            raise ValueError("Only completed SLM can be exported; use fit latest.pt to resume")
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(directory / "tokenizer")
        self.model.encoder.config.save_pretrained(directory / "encoder")
        _atomic_torch_save(directory / "model.pt", self.model.state_dict())
        atomic_json(directory / "scope_slm.json", {"format_version": 1, "categories": self.categories,
            "config": self.config, "training": self.training, "metadata": self.metadata})

    @classmethod
    def load(cls, checkpoint, device="cuda"):
        directory = Path(checkpoint).expanduser().resolve()
        saved = json.loads((directory / "scope_slm.json").read_text())
        if saved["format_version"] != 1 or saved["training"].get("status") != "complete":
            raise ValueError("Not a completed supported SCOPE SLM checkpoint")
        obj = cls.__new__(cls)
        obj._portable_config = AutoConfig.from_pretrained(directory / "encoder", local_files_only=True)
        cfg = {**saved["config"], "device": device, "model_path": str(directory / "tokenizer"),
               "mixed_precision": saved["config"]["mixed_precision"] if device.startswith("cuda") else "no"}
        obj.__init__(saved["categories"], cfg)
        obj.model.load_state_dict(torch.load(directory / "model.pt", map_location=obj.device, weights_only=True))
        obj.trained, obj.training, obj.metadata = True, saved["training"], saved["metadata"]
        obj.metadata["loaded_device"] = device
        obj.model.eval()
        return obj
