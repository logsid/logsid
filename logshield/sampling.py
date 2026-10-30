"""Load the checksum-verified, content-only entropy sampler."""
import ast
import json
from pathlib import Path
from .io import DEFAULT_CACHE, ROOT, file_sha256

NATIVE_FILE = "unleash/sampling/entropy_sampling.py"


def native_source():
    manifest = json.loads((ROOT / "logshield" / "sampling_source.json").read_text())
    path = DEFAULT_CACHE / "upstream" / ("UNLEASH-" + manifest["commit"]) / NATIVE_FILE
    expected = manifest["source_files"][NATIVE_FILE]["sha256"]
    if file_sha256(path) != expected:
        raise ValueError("UNLEASH sampling source differs from the pinned official file")
    return path, expected, manifest["commit"]


def load_native_sampler(path, expected_sha):
    """Execute unchanged author functions; omit dataset metadata and script I/O."""
    path = Path(path)
    if file_sha256(path) != expected_sha:
        raise ValueError("UNLEASH sampling source checksum mismatch")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
                 and not (isinstance(node, ast.ImportFrom) and node.level)]
    scope = {"__name__": "logsid_pinned_unleash_sampling", "__file__": str(path)}
    exec(compile(tree, str(path), "exec"), scope)
    return scope
