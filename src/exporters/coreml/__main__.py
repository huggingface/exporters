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

import warnings

from argparse import ArgumentParser
from pathlib import Path

from coremltools import ComputeUnit
from coremltools.models import MLModel
from coremltools.models.utils import _is_macos, _macos_version

from transformers.models.auto import AutoFeatureExtractor, AutoProcessor, AutoTokenizer
from transformers.onnx.utils import get_preprocessor

from .convert import export
from .features import FeaturesManager
from .validate import validate_model_outputs
from ..utils import logging


def convert_model(preprocessor, model, model_coreml_config, args, use_past=False, seq2seq=None):
    coreml_config = model_coreml_config(model.config, use_past=use_past, seq2seq=seq2seq)

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

    filename = args.output
    if seq2seq == "encoder":
        filename = filename.parent / ("encoder_" + filename.name)
    elif seq2seq == "decoder":
        filename = filename.parent / ("decoder_" + filename.name)
    filename = filename.as_posix()

    mlmodel.save(filename)

    if args.atol is None:
        args.atol = coreml_config.atol_for_validation

    if not _is_macos() or _macos_version() < (12, 0):
        logger.info("Skipping model validation, requires macOS 12.0 or later")
    else:
        # Run validation on CPU
        mlmodel = MLModel(filename, compute_units=ComputeUnit.CPU_ONLY)
        validate_model_outputs(coreml_config, preprocessor, model, mlmodel, args.atol)

    logger.info(f"All good, model saved at: {filename}")


def main():
    parser = ArgumentParser("Hugging Face Transformers Core ML exporter")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    parser.add_argument(
        "--feature",
        choices=list(FeaturesManager.AVAILABLE_FEATURES_INCLUDING_LEGACY),
        default="feature-extraction",
        help="The type of features to export the model with.",
    )
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute difference tolerence when validating the model."
    )
    parser.add_argument(
        "--use_past", action="store_true", help="Export the model with precomputed hidden states (key and values in the attention blocks) for fast autoregressive decoding."
    )
    parser.add_argument(
        "--framework", type=str, choices=["pt", "tf"], default="pt", help="The framework to use for the Core ML export."
    )
    parser.add_argument(
        "--quantize", type=str, choices=["float32", "float16"], default="float16", help="Quantization option for the model weights."
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
    
    # Support legacy task names in CLI only
    feature = args.feature
    args.feature = FeaturesManager.map_from_synonym(args.feature)
    if feature != args.feature:
        deprecation_message = f"Feature '{feature}' is deprecated, please use '{args.feature}' instead."
        warnings.warn(deprecation_message, FutureWarning)

    # Allocate the model
    model = FeaturesManager.get_model_from_feature(
        args.feature, args.model, framework=args.framework, #cache_dir=args.cache_dir
    )
    model_kind, model_coreml_config = FeaturesManager.check_supported_model_or_raise(model, feature=args.feature)

    if args.feature in ["text2text-generation", "speech-seq2seq"]:
        logger.info(f"Converting encoder model...")

        convert_model(
            preprocessor,
            model,
            model_coreml_config,
            args,
            use_past=False,
            seq2seq="encoder"
        )

        logger.info(f"Converting decoder model...")

        convert_model(
            preprocessor,
            model,
            model_coreml_config,
            args,
            use_past=args.use_past,
            seq2seq="decoder"
        )
    else:
        convert_model(
            preprocessor,
            model,
            model_coreml_config,
            args,
            use_past=args.use_past,
        )


if __name__ == "__main__":
    logger = logging.get_logger("exporters.coreml")  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    main()
