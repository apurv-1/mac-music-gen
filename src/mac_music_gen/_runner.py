"""Start the GPU worker, watch it, and turn its outcome into a result or an exception."""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from .errors import EngineError, EngineNotInstalled, GpuUnavailable, WeightsMissing

# Tests point this at a fake worker script.
WORKER_ARGV: list[str] = [sys.executable, "-m", "mac_music_gen.acestep_worker"]
TAIL_LINES = 40

_KINDS = {
    "gpu_unavailable": GpuUnavailable,
    "engine_not_installed": EngineNotInstalled,
    "weights_missing": WeightsMissing,
}


def _pump(stream, tail: collections.deque, verbose: bool) -> None:
    for line in stream:
        tail.append(line.rstrip("\n"))
        if verbose:
            print(line, end="", file=sys.stderr)


def _tail_text(tail: collections.deque) -> str:
    return ("\nlast worker output:\n" + "\n".join(tail)) if tail else ""


def run_worker(
    request: dict, job_dir: Path, env: dict[str, str], *, timeout: float | None = None, verbose: bool = False
) -> dict:
    """Run one request in a fresh worker process and return its parsed result.

    The worker's own logs are shown only with verbose; on failure the last lines come back
    inside the exception, so a crash is never silent.
    """
    job_dir = Path(job_dir)
    (job_dir / "request.json").write_text(json.dumps(request))
    proc = subprocess.Popen(
        [*WORKER_ARGV, str(job_dir)],
        env={**os.environ, **env},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    tail: collections.deque = collections.deque(maxlen=TAIL_LINES)
    reader = threading.Thread(target=_pump, args=(proc.stderr, tail, verbose), daemon=True)
    reader.start()
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        raise EngineError(f"the worker did not finish within {timeout:g} s and was stopped{_tail_text(tail)}") from None
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise
    reader.join(timeout=5)

    result_path = job_dir / "result.json"
    if not result_path.is_file():
        raise EngineError(f"the worker exited with code {code} without a result{_tail_text(tail)}")
    result = json.loads(result_path.read_text())
    if not result.get("ok"):
        kind = result.get("error_kind") or "engine"
        message = result.get("error") or "the worker failed"
        if kind in _KINDS:
            raise _KINDS[kind](message)
        raise EngineError(f"{message}{_tail_text(tail)}")
    return result


def probe(timeout: float = 300) -> dict:
    """Ask a fresh worker what the GPU stack can see."""
    try:
        done = subprocess.run([*WORKER_ARGV, "--probe"], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"engine_installed": False, "error": f"the probe did not finish within {timeout:g} s"}
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"engine_installed": False, "error": (done.stderr.strip().splitlines() or ["probe failed"])[-1]}
