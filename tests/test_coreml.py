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

import pytest

from unittest import TestCase
from parameterized import parameterized
from transformers import AutoConfig, is_tf_available, is_torch_available

from exporters.coreml import (
    CoreMLConfig,
    export,
    validate_model_outputs,
)
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import require_tf, require_torch, require_vision, slow
from .testing_utils import require_coreml, require_macos


if is_torch_available() or is_tf_available():
    from exporters.coreml.features import FeaturesManager


class TextCoreMLConfig(CoreMLConfig):
    modality = "text"


class CoreMLConfigTestCase(TestCase):
    def test_unknown_modality(self):
        with pytest.raises(ValueError):
            config = CoreMLConfig(None, task="feature-extraction")

    def test_unknown_task(self):
        with pytest.raises(AssertionError):
            config = TextCoreMLConfig(None, task="unknown-task")
            _ = config.inputs

    def test_sequence_length(self):
        config = TextCoreMLConfig(None, task="feature-extraction")
        flexible_outputs = config.get_flexible_outputs()
        self.assertEqual(len(flexible_outputs), 1)
        self.assertIn("last_hidden_state", flexible_outputs)

        flexible_output = flexible_outputs["last_hidden_state"]
        self.assertEqual(len(flexible_output), 1)
        self.assertEqual(flexible_output[0]["axis"], 1)
        self.assertEqual(flexible_output[0]["min"], 1)
        self.assertEqual(flexible_output[0]["max"], config.max_sequence_length)

        config = TextCoreMLConfig(None, task="text-classification")
        flexible_outputs = config.get_flexible_outputs()
        self.assertTrue(len(flexible_outputs) == 0)


PYTORCH_EXPORT_MODELS = {
    ("beit", "microsoft/beit-base-patch16-224"),
    ("bert", "bert-base-cased"),
    ("convnext", "facebook/convnext-tiny-224"),
    ("cvt", "microsoft/cvt-21-384-22k"),
    ("distilbert", "distilbert-base-cased"),
    ("gpt2", "distilgpt2"),
    ("levit", "facebook/levit-128S"),
    ("mobilebert", "google/mobilebert-uncased"),
    ("mobilevit", "apple/mobilevit-small"),
    ("mobilevitv2", "apple/mobilevitv2-1.0-imagenet1k-256"),
    ("segformer", "nvidia/mit-b0"),
    ("squeezebert", "squeezebert/squeezebert-uncased"),
    ("t5", "t5-small"),
    ("vit", "google/vit-base-patch16-224"),
    ("yolos", "hustvl/yolos-tiny"),
}

PYTORCH_EXPORT_WITH_PAST_MODELS = {
    ("ctrl", "sshleifer/tiny-ctrl"),
    #TODO ("gpt2", "distilgpt2"),
}

PYTORCH_EXPORT_SEQ2SEQ_WITH_PAST_MODELS = {}

TENSORFLOW_EXPORT_DEFAULT_MODELS = {}

TENSORFLOW_EXPORT_WITH_PAST_MODELS = {}

TENSORFLOW_EXPORT_SEQ2SEQ_WITH_PAST_MODELS = {}


# Copied from tests.onnx.test_onnx_v2._get_models_to_test
def _get_models_to_test(export_models_list):
    models_to_test = []
    if is_torch_available() or is_tf_available():
        for name, model, *features in export_models_list:
            if features:
                feature_config_mapping = {
                    feature: FeaturesManager.get_config(name, feature) for _ in features for feature in _
                }
            else:
                feature_config_mapping = FeaturesManager.get_supported_features_for_model_type(name)

            for feature, coreml_config_class_constructor in feature_config_mapping.items():
                models_to_test.append((f"{name}_{feature}", name, model, feature, coreml_config_class_constructor))
        return sorted(models_to_test)
    else:
        # Returning some dummy test that should not be ever called because of the @require_torch / @require_tf
        # decorators.
        # The reason for not returning an empty list is because parameterized.expand complains when it's empty.
        return [("dummy", "dummy", "dummy", "dummy", CoreMLConfig.from_model_config)]


@require_coreml
@require_macos
class CoreMLExportTestCase(TestCase):
    """
    Integration tests ensuring supported models are correctly exported
    """

    def _coreml_export(self, test_name, name, model_name, feature, coreml_config_class_constructor):
        model_class = FeaturesManager.get_model_class_for_feature(feature)
        config = AutoConfig.from_pretrained(model_name)
        model = model_class.from_config(config)
        coreml_config = coreml_config_class_constructor(model.config)
        preprocessor = get_preprocessor(model_name)

        try:
            if feature in ["text2text-generation", "speech-seq2seq"]:
                coreml_config.seq2seq = "encoder"
                mlmodel = export(
                    preprocessor,
                    model,
                    coreml_config,
                    quantize="float32",
                )
                validate_model_outputs(
                    coreml_config,
                    preprocessor,
                    model,
                    mlmodel,
                    coreml_config.atol_for_validation,
                )

                coreml_config.seq2seq = "decoder"
                mlmodel = export(
                    preprocessor,
                    model,
                    coreml_config,
                    quantize="float32",
                )
                validate_model_outputs(
                    coreml_config,
                    preprocessor,
                    model,
                    mlmodel,
                    coreml_config.atol_for_validation,
                )
            else:
                mlmodel = export(
                    preprocessor,
                    model,
                    coreml_config,
                    quantize="float32",
                )

                validate_model_outputs(
                    coreml_config,
                    preprocessor,
                    model,
                    mlmodel,
                    coreml_config.atol_for_validation,
                )
        except (RuntimeError, ValueError) as e:
            self.fail(f"{name}, {feature} -> {e}")

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_MODELS))
    @slow
    @require_torch
    @require_vision
    def test_pytorch_export(self, test_name, name, model_name, feature, coreml_config_class_constructor):
        self._coreml_export(test_name, name, model_name, feature, coreml_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_WITH_PAST_MODELS), skip_on_empty=True)
    @slow
    @require_torch
    def test_pytorch_export_with_past(self, test_name, name, model_name, feature, coreml_config_class_constructor):
        self._coreml_export(test_name, name, model_name, feature, coreml_config_class_constructor)

    @parameterized.expand(_get_models_to_test(PYTORCH_EXPORT_SEQ2SEQ_WITH_PAST_MODELS), skip_on_empty=True)
    @slow
    @require_torch
    def test_pytorch_export_seq2seq_with_past(
        self, test_name, name, model_name, feature, coreml_config_class_constructor
    ):
        self._coreml_export(test_name, name, model_name, feature, coreml_config_class_constructor)

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_DEFAULT_MODELS), skip_on_empty=True)
    @slow
    @require_tf
    @require_vision
    def test_tensorflow_export(self, test_name, name, model_name, feature, coreml_config_class_constructor):
        self._coreml_export(test_name, name, model_name, feature, coreml_config_class_constructor)

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_WITH_PAST_MODELS), skip_on_empty=True)
    @slow
    @require_tf
    def test_tensorflow_export_with_past(self, test_name, name, model_name, feature, coreml_config_class_constructor):
        self._coreml_export(test_name, name, model_name, feature, coreml_config_class_constructor)

    @parameterized.expand(_get_models_to_test(TENSORFLOW_EXPORT_SEQ2SEQ_WITH_PAST_MODELS), skip_on_empty=True)
    @slow
    @require_tf
    def test_tensorflow_export_seq2seq_with_past(
        self, test_name, name, model_name, feature, coreml_config_class_constructor
    ):
        self._coreml_export(test_name, name, model_name, feature, coreml_config_class_constructor)
