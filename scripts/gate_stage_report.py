#!/usr/bin/env python3
"""Report a failing gate stage: print ids/summary + tail, copy the full log."""
import argparse, os, re, shutil, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--copy-to", required=True)
    args = ap.parse_args()
    log_path = Path(args.log)
    dest_dir = Path(args.copy_to)
    dest_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", args.stage.strip())
    run_log = dest_dir / (slug + ".log")
    if log_path.is_file():
        shutil.copyfile(log_path, run_log)
    text = log_path.read_text(errors="replace") if log_path.is_file() else ""
    lines = text.splitlines()
    id_lines = [ln for ln in lines
                if re.match(r"^(FAILED|ERROR) ", ln)
                or "failed," in ln or "passed," in ln
                or "error during collection" in ln]
    if id_lines:
        print("--- failing ids / summary ---")
        for ln in id_lines[-40:]:
            print("    " + ln)
    total = len(lines)
    printed_ids = min(len(id_lines), 40)
    printed = printed_ids + 25
    withheld = total - printed if total > printed else 0
    if withheld > 0:
        print(f"    ... {withheld} more lines withheld (full log: {run_log})")
    print("    --- tail ---")
    for ln in lines[-25:]:
        print("    " + ln)
    print(f"    full log: {run_log}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
