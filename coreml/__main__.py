# coding=utf-8
# Copyright 2021-2022 The HuggingFace Team. All rights reserved.
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

from argparse import ArgumentParser
from pathlib import Path

from coremltools import ComputeUnit
from coremltools.models.utils import _is_macos, _macos_version

from transformers.models.auto import AutoFeatureExtractor, AutoProcessor, AutoTokenizer
from transformers.onnx.utils import get_preprocessor
from transformers.utils import logging
from .convert import export
from .features import FeaturesManager
from .validate import validate_model_outputs


def main():
    parser = ArgumentParser("Hugging Face Transformers Core ML exporter")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    parser.add_argument(
        "--feature",
        choices=list(FeaturesManager.AVAILABLE_FEATURES),
        default="default",
        help="The type of features to export the model with.",
    )
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute difference tolerence when validating the model."
    )
    parser.add_argument(
        "--framework", type=str, choices=["pt", "tf"], default="pt", help="The framework to use for the Core ML export."
    )
    parser.add_argument(
        "--quantize", type=str, choices=["float32", "float16"], default="float32", help="Quantization option for the model weights."
    )
    parser.add_argument(
        "--compute_units", type=str, choices=["all", "cpu_and_gpu", "cpu_only", "cpu_and_ne"], default="all", help="Optimize the model for CPU, GPU, and/or Neural Engine."
    )
    # parser.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    parser.add_argument(
        "--preprocessor",
        type=str,
        choices=["auto", "tokenizer", "feature_extractor", "processor"],
        default="auto",
        help="Which type of preprocessor to use. 'auto' tries to automatically detect it.",
    )
    parser.add_argument("output", type=Path, help="Path indicating where to store generated Core ML model.")

    args = parser.parse_args()

    if (not args.output.is_file()) and (args.output.suffix not in [".mlpackage", ".mlmodel"]):
        args.output = args.output.joinpath("Model.mlpackage")
    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    # Instantiate the appropriate preprocessor
    if args.preprocessor == "auto":
        preprocessor = get_preprocessor(args.model)
    elif args.preprocessor == "tokenizer":
        preprocessor = AutoTokenizer.from_pretrained(args.model)
    elif args.preprocessor == "feature_extractor":
        preprocessor = AutoFeatureExtractor.from_pretrained(args.model)
    elif args.preprocessor == "processor":
        preprocessor = AutoProcessor.from_pretrained(args.model)
    else:
        raise ValueError(f"Unknown preprocessor type '{args.preprocessor}'")

    # Allocate the model
    model = FeaturesManager.get_model_from_feature(
        args.feature, args.model, framework=args.framework, #cache_dir=args.cache_dir
    )
    model_kind, model_coreml_config = FeaturesManager.check_supported_model_or_raise(model, feature=args.feature)
    coreml_config = model_coreml_config(model.config)

    compute_units = ComputeUnit.ALL
    if args.compute_units == "cpu_and_gpu":
        compute_units = ComputeUnit.CPU_AND_GPU
    elif args.compute_units == "cpu_only":
        compute_units = ComputeUnit.CPU_ONLY
    elif args.compute_units == "cpu_and_ne":
        compute_units = ComputeUnit.CPU_AND_NE

    mlmodel = export(
        preprocessor,
        model,
        coreml_config,
        quantize=args.quantize,
        compute_units=compute_units,
    )

    mlmodel.save(args.output.as_posix())

    if args.atol is None:
        args.atol = coreml_config.atol_for_validation

    if not _is_macos() or _macos_version() < (12, 0):
        logger.info("Skipping model validation, requires macOS 12.0 or later")
    else:
        validate_model_outputs(coreml_config, preprocessor, model, mlmodel, args.atol)

    logger.info(f"All good, model saved at: {args.output.as_posix()}")


if __name__ == "__main__":
    logger = logging.get_logger("exporters.coreml")  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    main()
