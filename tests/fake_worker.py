"""Stand-in for mac_music_gen.acestep_worker, driven by FAKE_WORKER_MODE. Used by the tests only."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def main(argv):
    mode = os.environ.get("FAKE_WORKER_MODE", "ok")
    if argv == ["--probe"]:
        print(
            json.dumps(
                {
                    "engine_installed": True,
                    "mps": mode != "gpu",
                    "mlx_metal": mode != "gpu",
                    "torch": "fake",
                    "mlx": "fake",
                    "macos": "26.0",
                }
            )
        )
        return 0
    job_dir = Path(argv[0])
    request = json.loads((job_dir / "request.json").read_text())
    print("loading the fake model", file=sys.stderr)
    if mode == "hang":
        time.sleep(60)
    if mode == "crash":
        print("Metal abort: fake crash", file=sys.stderr)
        return 134
    if mode in ("gpu", "engine", "engine_not_installed"):
        kind = {"gpu": "gpu_unavailable"}.get(mode, mode)
        (job_dir / "result.json").write_text(json.dumps({"ok": False, "error_kind": kind, "error": f"fake {mode}"}))
        return 1
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise SystemExit("offline mode was not passed to the worker")
    jobs = []
    for i, job in enumerate(request["jobs"]):
        sr = 48000
        t = np.arange(int(job["seconds"] * sr)) / sr
        x = 0.3 * np.sin(2 * np.pi * (220 + 10 * job["seed"] % 50) * t)
        path = job_dir / f"raw-{i}.wav"
        sf.write(str(path), np.stack([x, x], axis=1).astype(np.float32), sr, subtype="FLOAT")
        jobs.append({"raw_path": str(path), "sample_rate": sr, "seed": job["seed"], "generate_s": 2.0})
    (job_dir / "request-seen.json").write_text(json.dumps(request))
    result = {
        "ok": True,
        "error_kind": None,
        "error": None,
        "load_s": 1.0,
        "jobs": jobs,
        "runtime": {"device": "mps", "mlx_dit": True, "mlx_vae": True, "text_encoder_device": "mps"},
    }
    (job_dir / "result.json").write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
