# Copyright 2026 The Torch-Spyre Authors.
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

import sys

import click

from .core import launch_from_cli


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-i",
    "--input",
    multiple=True,
    metavar="FILE",
    type=click.Path(exists=True, dir_okay=False),
    help="input tensor file (.pt or .bin), in arg_index order",
)
@click.option(
    "-o",
    "--output",
    multiple=True,
    metavar="FILE",
    type=click.Path(exists=True, dir_okay=False),
    help="EXPECTED output; the launch result is compared against it",
)
@click.option(
    "--bind",
    multiple=True,
    metavar="SYM=N",
    help="bind a symbolic dimension, e.g. --bind s0=128",
)
@click.option("--rtol", type=float, default=None, help="relative tolerance")
@click.option("--atol", type=float, default=None, help="absolute tolerance")
@click.option("--exact", is_flag=True, help="require bitwise equality")
@click.option(
    "--save-output",
    metavar="FILE",
    default=None,
    help="write the device result here as a .pt",
)
@click.argument("path", default=".")
def launch(path, input, output, bind, rtol, atol, exact, save_output):
    """Launch SpyreCode, and check the answer when given a reference.

    With no files, the tensors come from the folder's launch_spec.json with
    ones for inputs, and the outputs are printed.

    
      spyre launch <folder>
      spyre launch -i a.pt -i b.pt <folder>
      spyre launch -i a.pt -i b.pt -o expected.pt <folder>

    Exit codes: 0 matched (or nothing to compare), 1 mismatch, 2 could not
    launch -- a mismatch and a malformed invocation are different failures.
    """
    bindings = {}
    for item in bind:
        sym, sep, value = item.partition("=")
        if not sep or not value.isdigit():
            raise click.BadParameter(f"expected --bind SYM=N, got {item!r}")
        bindings[sym] = int(value)

    try:
        matched = launch_from_cli(
            path,
            list(input),
            list(output),
            bindings,
            rtol=rtol,
            atol=atol,
            exact=exact,
            save_output=save_output,
        )
    except (ValueError, OSError) as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(2)
    sys.exit(0 if matched else 1)


def main():
    cli()
