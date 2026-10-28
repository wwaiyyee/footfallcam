"""Explicit one-time weight download; normal inference is local-only."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


MODEL_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["n", "s", "m"], default="s")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"models/yolo11{args.variant}.pt")
    model_url = f"{MODEL_BASE_URL}/yolo11{args.variant}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".json")
    if not output.exists():
        temporary = output.with_suffix(".part")
        urllib.request.urlretrieve(model_url, temporary)
        temporary.replace(output)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["sha256"] != checksum:
            raise ValueError("Model checksum changed. Restore the original model or review the replacement explicitly.")
    manifest = {"name": f"YOLO11{args.variant}", "filename": output.name, "source": model_url,
                "sha256": checksum, "bytes": output.stat().st_size,
                "license_note": "Ultralytics AGPL-3.0 / enterprise; review licensing before redistribution."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
