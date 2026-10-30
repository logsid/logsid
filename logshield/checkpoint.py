"""Pinned backbone identity and reproducible checkpoint helpers."""
import os
import random
from pathlib import Path

COMMIT = "0a6dd3239668adae6b1160eeea5543034fb21849"
MODEL_ID = "FacebookAI/roberta-base"
MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


def _rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(path, payload):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
