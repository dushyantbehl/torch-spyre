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

"""Compare DXP debug artifacts against a golden sample.

Usage:
    python3 compare_golden.py --config golden_config.yaml
"""

import argparse
import difflib
import glob
import os
import sys

import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def kernel_prefix(dirname):
    """Strip the random tempdir suffix to get the kernel name prefix.

    e.g. 'sdsc_fused_gelu_0_7ha4yazj' -> 'sdsc_fused_gelu_0'
    """
    parts = dirname.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else dirname


def find_kernel_dirs(example_dir):
    """Return a dict mapping kernel prefix -> full directory path."""
    if not os.path.isdir(example_dir):
        return {}
    result = {}
    for entry in os.scandir(example_dir):
        if entry.is_dir():
            prefix = kernel_prefix(entry.name)
            result[prefix] = entry.path
    return result


def collect_files(kernel_dir, patterns):
    """Glob for files matching patterns relative to kernel_dir."""
    files = set()
    for pattern in patterns:
        matches = glob.glob(os.path.join(kernel_dir, pattern), recursive=True)
        for m in matches:
            files.add(os.path.relpath(m, kernel_dir))
    return sorted(files)


def diff_files(golden_path, current_path):
    """Return unified diff lines between two files, or None if they match."""
    with open(golden_path) as f:
        golden_lines = f.readlines()
    with open(current_path) as f:
        current_lines = f.readlines()

    diff = list(
        difflib.unified_diff(
            golden_lines,
            current_lines,
            fromfile=f"golden: {golden_path}",
            tofile=f"current: {current_path}",
        )
    )
    return diff if diff else None


def compare_file_sets(label, golden_kdir, current_kdir, prefix, patterns):
    """Compare files matching patterns between two kernel dirs. Returns (passed, failed)."""
    golden_files = collect_files(golden_kdir, patterns)
    current_files = collect_files(current_kdir, patterns)

    if not golden_files and not current_files:
        return 0, 0

    passed = 0
    failed = 0
    all_files = sorted(set(golden_files) | set(current_files))

    for rel_path in all_files:
        gf = os.path.join(golden_kdir, rel_path)
        cf = os.path.join(current_kdir, rel_path)
        tag = f"{prefix}/{rel_path}"

        if not os.path.exists(gf):
            print(f"  NEW    [{label}]: {tag} (not in golden)")
            failed += 1
        elif not os.path.exists(cf):
            print(f"  MISSING [{label}]: {tag} (not in current)")
            failed += 1
        else:
            diff = diff_files(gf, cf)
            if diff is None:
                print(f"  PASS   [{label}]: {tag}")
                passed += 1
            else:
                print(f"  FAIL   [{label}]: {tag}")
                for line in diff:
                    print(f"    {line}", end="")
                print()
                failed += 1

    return passed, failed


def compare_example(example_name, golden_dir, current_dir, input_patterns, output_patterns):
    """Compare all matching kernel dirs for one example. Returns (passed, failed) counts."""
    golden_kernels = find_kernel_dirs(golden_dir)
    current_kernels = find_kernel_dirs(current_dir)

    if not golden_kernels:
        print(f"  WARNING: no kernel dirs found in golden {golden_dir}")
        return 0, 0
    if not current_kernels:
        print(f"  WARNING: no kernel dirs found in current {current_dir}")
        return 0, len(golden_kernels)

    passed = 0
    failed = 0

    for prefix, golden_kdir in sorted(golden_kernels.items()):
        if prefix not in current_kernels:
            print(f"  MISSING: {prefix} not found in current run")
            failed += 1
            continue

        current_kdir = current_kernels[prefix]

        # Compare input SDSCs
        p, f = compare_file_sets("input", golden_kdir, current_kdir, prefix, input_patterns)
        passed += p
        failed += f

        # Compare debug output
        p, f = compare_file_sets("output", golden_kdir, current_kdir, prefix, output_patterns)
        passed += p
        failed += f

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Compare DXP artifacts against golden sample")
    parser.add_argument(
        "--config",
        default="golden_config.yaml",
        help="Path to config YAML (default: golden_config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    golden_base = config["golden"]["collected_dir"]
    current_base = config["current"]["collected_dir"]
    method = config["compare"].get("method", "diff")
    input_patterns = config["compare"].get(
        "input_files", ["execute/*/sdsc.json", "sdsc_*.json"]
    )
    output_patterns = config["compare"].get(
        "output_files",
        ["execute/*/debug/**/*.json", "execute/*/debug/**/*.txt",
         "debug/**/*.json", "debug/**/*.txt"],
    )

    if method != "diff":
        print(f"ERROR: unsupported compare method '{method}' (only 'diff' supported)")
        sys.exit(2)

    print(f"Golden:  {golden_base} (torch-spyre {config['golden']['torch_spyre_version']}, "
          f"deeptools {config['golden']['deeptools_version']})")
    print(f"Current: {current_base} (torch-spyre {config['current']['torch_spyre_version']}, "
          f"deeptools {config['current']['deeptools_version']})")
    print(f"Method:  {method}")
    print(f"Input patterns:  {input_patterns}")
    print(f"Output patterns: {output_patterns}")
    print()

    if not os.path.isdir(golden_base):
        print(f"ERROR: golden directory '{golden_base}' does not exist")
        sys.exit(2)
    if not os.path.isdir(current_base):
        print(f"ERROR: current directory '{current_base}' does not exist")
        sys.exit(2)

    total_passed = 0
    total_failed = 0

    golden_examples = sorted(
        e.name for e in os.scandir(golden_base) if e.is_dir()
    )

    for example in golden_examples:
        golden_dir = os.path.join(golden_base, example)
        current_dir = os.path.join(current_base, example)

        print(f"=== {example} ===")
        if not os.path.isdir(current_dir):
            print(f"  MISSING: example '{example}' not in current run")
            total_failed += 1
            print()
            continue

        p, f = compare_example(example, golden_dir, current_dir, input_patterns, output_patterns)
        total_passed += p
        total_failed += f
        print()

    print("=" * 40)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
