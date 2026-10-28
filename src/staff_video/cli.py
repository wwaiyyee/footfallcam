from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import ROOT, load_config
from .pipeline import analyze_video
from .video import inspect_video


def main(argv=None):
    parser = argparse.ArgumentParser(description="Identify tag-confirmed staff with a local, shared video pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Decode the first frame and inspect video metadata without loading a model.")
    inspect.add_argument("video", type=Path)
    analyze = commands.add_parser("analyze", help="Process every decoded frame and export video, CSV, JSON, examples, and ZIP.")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--config", type=Path, default=ROOT / "config/default.yaml")
    analyze.add_argument("--output", type=Path, default=Path("outputs"))
    analyze.add_argument("--model", type=Path)
    analyze.add_argument("--tag-reference", type=Path)
    analyze.add_argument("--device", choices=["cpu", "mps", "0"])
    analyze.add_argument("--tag-threshold", type=float)
    analyze.add_argument("--labels", type=Path, help="Optional frame_index,staff_present CSV for real evaluation.")
    analyze.add_argument("--person-labels", type=Path, help="Optional complete person-box labels for selected frames; scores person/staff/tag precision and recall.")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_video(args.video).to_dict(), indent=2))
            return 0
        config = load_config(args.config)
        changes = {}
        for name, value in [("model_path", args.model), ("tag_reference", args.tag_reference),
                            ("device", args.device), ("tag_threshold", args.tag_threshold)]:
            if value is not None:
                changes[name] = str(value.resolve()) if isinstance(value, Path) else value
        config = config.changed(**changes)
        labels = None
        person_labels = None
        if args.labels:
            import pandas as pd

            labels = pd.read_csv(args.labels)
        if args.person_labels:
            import pandas as pd

            person_labels = pd.read_csv(args.person_labels)
        last_stage = [None]

        def progress(event):
            if event["stage"] != last_stage[0] or event["processed_frames"] % 100 == 1:
                print(f'{event["stage"]}: {event["processed_frames"]}/{event["total_frames"]} frames '
                      f'({event["elapsed_seconds"]:.1f}s)', file=sys.stderr, flush=True)
                last_stage[0] = event["stage"]

        result = analyze_video(args.video, args.output, config, progress, labels, person_labels)
        print(json.dumps({"output_directory": str(result.directory), **result.summary}, indent=2))
        return 0
    except Exception as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
