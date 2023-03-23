# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import unittest
import importlib.util
from transformers.utils.versions import importlib_metadata


_coreml_available = importlib.util.find_spec("coremltools") is not None
try:
    _coreml_version = importlib_metadata.version("coremltools")

    from coremltools.models.utils import _is_macos, _macos_version
    _macos_available = _is_macos() and _macos_version() >= (12, 0)

except importlib_metadata.PackageNotFoundError:
    _coreml_available = False
    _macos_available = False

def is_coreml_available():
    return _coreml_available

def is_macos_available():
    return _macos_available

def require_coreml(test_case):
    return unittest.skipUnless(is_coreml_available(), "test requires Core ML")(test_case)

def require_macos(test_case):
    return unittest.skipUnless(is_macos_available(), "test requires macOS")(test_case)
