#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run all examples and collect SDSC + DXP debug artifacts.

Usage:
    python3 collect.py --output collected_golden
    python3 collect.py --output collected_new
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time


def get_spyre_cache():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from torch._inductor.runtime.runtime_utils import cache_dir; print(cache_dir())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return os.path.join(result.stdout.strip(), "inductor-spyre")


def main():
    parser = argparse.ArgumentParser(description="Collect SDSC + DXP debug artifacts")
    parser.add_argument(
        "--output",
        default="collected",
        help="Output directory for collected artifacts (default: collected)",
    )
    args = parser.parse_args()

    examples_dir = os.path.join(os.path.dirname(__file__), "examples")
    collect_dir = os.path.abspath(args.output)
    examples = sorted(glob.glob(os.path.join(examples_dir, "*.py")))

    if not examples:
        print("No examples found in examples/")
        sys.exit(1)

    spyre_cache = get_spyre_cache()
    env = {**os.environ, "DXP_DEBUG": "1"}

    print(f"Spyre cache: {spyre_cache}")
    print(f"Collecting to: {collect_dir}")
    print(f"Found {len(examples)} examples\n")

    for example in examples:
        name = os.path.splitext(os.path.basename(example))[0]
        os.makedirs(spyre_cache, exist_ok=True)
        before = time.time()

        print(f"--- Running {name} ---")
        result = subprocess.run(
            [sys.executable, example],
            env=env,
        )
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}), skipping collection\n")
            continue

        dest = os.path.join(collect_dir, name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.makedirs(dest)

        count = 0
        if os.path.isdir(spyre_cache):
            for entry in os.scandir(spyre_cache):
                if entry.is_dir() and entry.stat().st_mtime > before:
                    kernel_dest = os.path.join(dest, os.path.basename(entry.path))
                    copied = False

                    # Copy input SDSCs: execute/*/sdsc.json
                    for sdsc_file in glob.glob(
                        os.path.join(entry.path, "execute", "*", "sdsc.json")
                    ):
                        rel = os.path.relpath(sdsc_file, entry.path)
                        dest_file = os.path.join(kernel_dest, rel)
                        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                        shutil.copy2(sdsc_file, dest_file)
                        copied = True

                    # Copy debug output: execute/*/debug/
                    for debug_dir in glob.glob(
                        os.path.join(entry.path, "execute", "*", "debug")
                    ):
                        rel = os.path.relpath(debug_dir, entry.path)
                        dest_debug = os.path.join(kernel_dest, rel)
                        shutil.copytree(debug_dir, dest_debug)
                        copied = True

                    # Also handle bundle mode: sdsc_*.json at top level
                    for sdsc_file in glob.glob(
                        os.path.join(entry.path, "sdsc_*.json")
                    ):
                        rel = os.path.relpath(sdsc_file, entry.path)
                        dest_file = os.path.join(kernel_dest, rel)
                        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                        shutil.copy2(sdsc_file, dest_file)
                        copied = True

                    # Bundle mode debug: debug/ at top level
                    top_debug = os.path.join(entry.path, "debug")
                    if os.path.isdir(top_debug):
                        dest_debug = os.path.join(kernel_dest, "debug")
                        if not os.path.exists(dest_debug):
                            shutil.copytree(top_debug, dest_debug)
                        copied = True

                    if copied:
                        count += 1

        print(f"  Collected {count} kernel dir(s) -> {dest}/\n")

    print("Done. Artifacts in collected/")


if __name__ == "__main__":
    main()
