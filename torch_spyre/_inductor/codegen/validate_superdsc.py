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

"""Validate a SuperDSC dict against the JSON Schema in superdsc_schema.json."""

import json
import os


_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "superdsc_schema.json")


def validate_superdsc(sdsc_dict):
    """Validate *sdsc_dict* against the SuperDSC JSON Schema.

    Prints whether validation passed or failed.  Never raises — if
    ``jsonschema`` is not installed the check is silently skipped.
    """
    try:
        import jsonschema
    except ImportError:
        print("WARNING: jsonschema not installed, skipping SuperDSC validation")
        return

    with open(_SCHEMA_PATH) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=sdsc_dict, schema=schema)
        print("SuperDSC validated successfully")
    except jsonschema.ValidationError as e:
        print(f"SuperDSC validation failed: {e.message}")
