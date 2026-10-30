<div align="center">

# LogSID

**Detecting Sensitive Information in Large-Scale Logs**

Anonymous research artifact

[![Build and Test](https://github.com/logsid/logsid/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/logsid/logsid/actions/workflows/build_and_test.yml)
[![Release](https://img.shields.io/github/v/release/logsid/logsid?color=1685df)](https://github.com/logsid/logsid/releases/latest)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
[![Dataset on Zenodo](https://img.shields.io/badge/Dataset-Zenodo-1685df)][dataset]

[Project page](https://logsid.github.io/) · [Quick start](#quick-start) · [Dataset](#dataset) · [Build and test](#build-and-test) · [Releases](https://github.com/logsid/logsid/releases)

</div>

LogSID detects sensitive character spans in log messages through a content-aware cache, a small span-prediction model, and selective LLM calls. Predictions preserve overlapping spans and boundaries within tokens.

## Overview

![LogSID pipeline: historical annotations initialize the cache and small language model; messages advance from the cache to the SLM and LLM when a prediction is not accepted; checked model outputs can update the cache.](figures/overview.png)

The cache selects span rules from the current message content. Messages without an accepted cache prediction proceed to the RoBERTa-based SLM, then to the LLM when needed. Checked model outputs can update the cache for subsequent messages.

## Setup

Use **Linux or WSL with Python 3.12**. The default installation uses PyTorch 2.7.0 with CUDA 12.8.

```bash
git clone https://github.com/logsid/logsid.git
cd logsid
bash scripts/setup.sh cu128
```

<details>
<summary><strong>CPU installation and interpreter selection</strong></summary>

```bash
bash scripts/setup.sh cpu
export LOGSID_DEVICE=cpu
```

Set `LOGSID_PYTHON` to a Python 3.12 executable when creating the environment, or to the environment's interpreter when running commands.

</details>

<details>
<summary><strong>Choose a cache location, including another disk on WSL</strong></summary>

Downloaded archives, indexes, model checkpoints, and predictions use `LOGSID_CACHE`, which defaults to `~/.cache/logsid-paper`. Set it before preparing data and keep it consistent across runs. For example, on WSL:

```bash
export LOGSID_CACHE=/mnt/d/LogSID-cache/logsid
```

The compressed dataset is about 1.5 GiB. Extracted data, indexes, models, and predictions need additional space.

</details>

## Quick start

Run one source first:

```bash
bash run.sh download --sources Apache
bash run.sh prepare --sources Apache
bash run.sh detect --sources Apache --read-key
```

| Command | What it does |
| --- | --- |
| `download` | Fetches the source archive from Zenodo, verifies SHA-256, and extracts it. |
| `prepare` | Builds data indexes and chronological order; fetches the pinned entropy sampler and public RoBERTa backbone. |
| `detect` | Trains the source-specific SLM, fits and calibrates the cache, then processes the log stream. |

The default LLM is `qwen/qwen3.8-27b` through OpenRouter. `--read-key` prompts for an API key without saving it; `OPENROUTER_API_KEY` can also be set in the environment. LLM calls use the provider's paid API. Use `--llm-model MODEL_ID` to select another model.

<details>
<summary><strong>Run all eight sources</strong></summary>

Omit `--sources`:

```bash
bash run.sh download
bash run.sh prepare
bash run.sh detect --read-key
```

</details>

<details>
<summary><strong>Prepare and train without LLM calls</strong></summary>

```bash
bash run.sh detect --sources Apache --probe-only
```

</details>

<details>
<summary><strong>Resume a run or retry an API failure</strong></summary>

Repeat the detection command with the same cache and output directories to resume its checkpoint. An unresolved LLM response stops the stream for inspection. After resolving an API failure, add `--retry-failed` to permit a retry. Full-source evaluation is produced after every message has a prediction.

</details>

## Dataset

The [anonymous Zenodo review draft][dataset] provides **32,983,908 messages across eight sources**: Apache, BGL, OpenSSH, HealthApp, AIT-mail, Zanbil, Mac, and Thunderbird. The taxonomy defines **18 categories**, of which 12 occur in the benchmark. No Zenodo account is required to use the sharing link.

| File in each source archive | Contents |
| --- | --- |
| `<Source>.log.gz` | Original raw log lines. |
| `<Source>_labels.jsonl.gz` | Records with `LineId`, `Timestamp`, `Content`, and `Labels`. |
| `taxonomy.csv`, `statistics.csv`, `templates.csv` | Category definitions and dataset metadata. |
| `THIRD_PARTY_NOTICES.txt` | Source-specific redistribution terms. |

Labels use zero-based Unicode character offsets in `Content`, with an exclusive end: `Content[start:end]`. Each span includes `start`, `end`, and `category`; overlaps and nested spans are retained.

BGL, Thunderbird, and Zanbil use numbered 32 MiB ZIP volumes. The download command joins them, checks the archive and extracted-file hashes, and extracts the data into `data/`. Repeat the command to resume an interrupted download.

<details>
<summary><strong>Content boundaries and download options</strong></summary>

For Zanbil and AIT-mail, `Content` preserves the complete physical line apart from its line terminator. BGL and Thunderbird preserve the complete original log body. Apache, Mac, OpenSSH, and HealthApp follow the publisher's structured `Content`, including upstream normalization. Original complete lines remain in the `.log.gz` files; annotation offsets always refer to `Content`.

```bash
bash run.sh download --verify-only
```

Completed ZIPs are retained in `$LOGSID_CACHE/downloads`; successfully joined volumes are removed. Use `--no-keep-archive` to remove ZIPs after extraction, or `--base-url URL` to use a mirror containing the same named ZIPs and volumes.

</details>

## Build and test

[GitHub Actions](https://github.com/logsid/logsid/actions/workflows/build_and_test.yml) installs the pinned CPU dependencies, checks Python and shell syntax, and runs the synthetic tests on Python 3.12. Tests cover dataset downloads, sampling, preparation, and the detection pipeline. They require no dataset download, model weights, or LLM API key.

Run the same test suite locally after setup:

```bash
bash run.sh test
```

<details>
<summary><strong>Command help</strong></summary>

```bash
bash run.sh --help
bash run.sh download --help
bash run.sh prepare --help
bash run.sh detect --help
```

</details>

## Repository layout

```text
logshield/                  Cache, SLM, LLM, training, and detection
scripts/                    Setup, dataset downloader, and file manifest
tests/                      Synthetic regression and integration tests
figures/                    Method illustration
.github/workflows/          Build and test workflow
requirements.txt            Pinned dependencies
run.sh                      Command-line entry point
```

Generated data is ignored by Git. Samples, model checkpoints, cache state, predictions, and metrics stay under `LOGSID_CACHE`. Per-source predictions and reports are written to `$LOGSID_CACHE/logsid-paper/output/`; model and cache state live under `$LOGSID_CACHE/logsid-paper/work/`.

## Releases

Tagged [source releases](https://github.com/logsid/logsid/releases) provide a stable snapshot of the implementation, setup and run commands, and tests. Datasets are hosted separately on [Zenodo][dataset]. Baseline implementations, trained weights, experimental results, and generated intermediates are excluded from this repository and its releases.

[dataset]: https://zenodo.org/records/23810581?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjUxNWVlNWI3LTE0Y2MtNGIxNC04NWZkLWZiNDI1OGZjZTU2YiIsImRhdGEiOnt9LCJyYW5kb20iOiI0ZGEwMjgxNDhhNjYyODFmOTJiODEyYmVmMzRjZTg3MyJ9.sz5FkmO9MRwItV7NOjViAqVTtTTAG2Ynva1y5sp1xWPKJe60H_i7qClJ-punZawMc1jUzKgJijh9wZiBHQJDZA
