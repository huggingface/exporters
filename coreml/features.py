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

from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union

# TODO: temporary hack
import exporters
#import transformers

from transformers import PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import logging
from .config import CoreMLConfig


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_available():
    from transformers.models.auto import (
        AutoModel,
        # AutoModelForCausalLM,
        AutoModelForImageClassification,
        # AutoModelForImageSegmentation,
        AutoModelForMaskedImageModeling,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForNextSentencePrediction,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        # AutoModelForSeq2SeqLM,
        AutoModelForSemanticSegmentation,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )
if is_tf_available():
    from transformers.models.auto import (
        TFAutoModel,
        # TFAutoModelForCausalLM,
        # TFAutoModelForMaskedLM,
        # TFAutoModelForMultipleChoice,
        # TFAutoModelForQuestionAnswering,
        # TFAutoModelForSeq2SeqLM,
        # TFAutoModelForSequenceClassification,
        # TFAutoModelForTokenClassification,
    )
if not is_torch_available() and not is_tf_available():
    logger.warning(
        "The Core ML export features are only supported for PyTorch or TensorFlow. You will not be able to export models"
        " without one of these libraries installed."
    )


def supported_features_mapping(
    *supported_features: str, coreml_config_cls: str = None
) -> Dict[str, Callable[[PretrainedConfig], CoreMLConfig]]:
    """
    Generate the mapping between supported the features and their corresponding CoreMLConfig for a given model.

    Args:
        *supported_features: The names of the supported features.
        coreml_config_cls: The CoreMLConfig full name corresponding to the model.

    Returns:
        The dictionary mapping a feature to an CoreMLConfig constructor.
    """
    if coreml_config_cls is None:
        raise ValueError("A CoreMLConfig class must be provided")

    # TODO: temporary hack
    import exporters.coreml.models
    config_cls = exporters.coreml
    #config_cls = transformers
    for attr_name in coreml_config_cls.split("."):
        if not hasattr(config_cls, attr_name): continue  #TODO: temporary hack
        config_cls = getattr(config_cls, attr_name)
    mapping = {}
    for feature in supported_features:
        if "-with-past" in feature:
            task = feature.replace("-with-past", "")
            mapping[feature] = partial(config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(config_cls.from_model_config, task=feature)

    return mapping


class FeaturesManager:
    _TASKS_TO_AUTOMODELS = {}
    _TASKS_TO_TF_AUTOMODELS = {}
    if is_torch_available():
        _TASKS_TO_AUTOMODELS = {
            "default": AutoModel,
            # "causal-lm": AutoModelForCausalLM,
            "image-classification": AutoModelForImageClassification,
            # "image-segmentation": AutoModelForImageSegmentation,
            "masked-im": AutoModelForMaskedImageModeling,
            "masked-lm": AutoModelForMaskedLM,
            "multiple-choice": AutoModelForMultipleChoice,
            "next-sentence-prediction": AutoModelForNextSentencePrediction,
            "object-detection": AutoModelForObjectDetection,
            "question-answering": AutoModelForQuestionAnswering,
            "semantic-segmentation": AutoModelForSemanticSegmentation,
            # "seq2seq-lm": AutoModelForSeq2SeqLM,
            "sequence-classification": AutoModelForSequenceClassification,
            "token-classification": AutoModelForTokenClassification,
        }
    if is_tf_available():
        _TASKS_TO_TF_AUTOMODELS = {
            # "default": TFAutoModel,
            # "causal-lm": TFAutoModelForCausalLM,
            # "masked-lm": TFAutoModelForMaskedLM,
            # "multiple-choice": TFAutoModelForMultipleChoice,
            # "question-answering": TFAutoModelForQuestionAnswering,
            # "seq2seq-lm": TFAutoModelForSeq2SeqLM,
            # "sequence-classification": TFAutoModelForSequenceClassification,
            # "token-classification": TFAutoModelForTokenClassification,
        }

    _SUPPORTED_MODEL_TYPE = {
        # BEiT cannot be used with the masked image modeling autoclass, so this feature is excluded here
        "beit": supported_features_mapping(
            "default",
            "image-classification",
            "semantic-segmentation",
            coreml_config_cls="models.beit.BeitCoreMLConfig"
        ),
        "bert": supported_features_mapping(
            "default",
            "masked-lm",
            #"causal-lm",
            "multiple-choice",
            "question-answering",
            "sequence-classification",
            "token-classification",
            coreml_config_cls="models.bert.BertCoreMLConfig",
        ),
        "convnext": supported_features_mapping(
            "default",
            "image-classification",
            coreml_config_cls="models.convnext.ConvNextCoreMLConfig",
        ),
        "cvt": supported_features_mapping(
            "default",
            "image-classification",
            coreml_config_cls="models.cvt.CvtCoreMLConfig",
        ),
        "distilbert": supported_features_mapping(
            "default",
            "masked-lm",
            "multiple-choice",
            "question-answering",
            "sequence-classification",
            "token-classification",
            coreml_config_cls="models.distilbert.DistilBertCoreMLConfig",
        ),
        "gpt2": supported_features_mapping(
            "default",
            #"default-with-past",
            #"causal-lm",
            #"causal-lm-with-past",
            "sequence-classification",
            "token-classification",
            coreml_config_cls="models.gpt2.GPT2CoreMLConfig",
        ),
        "levit": supported_features_mapping(
            "default", "image-classification", coreml_config_cls="models.levit.LevitCoreMLConfig"
        ),
        "mobilebert": supported_features_mapping(
            "default",
            "masked-lm",
            "multiple-choice",
            "next-sentence-prediction",
            "question-answering",
            "sequence-classification",
            "token-classification",
            coreml_config_cls="models.mobilebert.MobileBertCoreMLConfig",
        ),
        "mobilevit": supported_features_mapping(
            "default",
            "image-classification",
            "semantic-segmentation",
            coreml_config_cls="models.mobilevit.MobileViTCoreMLConfig",
        ),
        "segformer": supported_features_mapping(
            "default",
            "image-classification",
            "semantic-segmentation",
            coreml_config_cls="models.segformer.SegformerCoreMLConfig",
        ),
        "squeezebert": supported_features_mapping(
            "default",
            "masked-lm",
            "multiple-choice",
            "question-answering",
            "sequence-classification",
            "token-classification",
            coreml_config_cls="models.squeezebert.SqueezeBertCoreMLConfig",
        ),
        "vit": supported_features_mapping(
            "default", "image-classification", "masked-im", coreml_config_cls="models.vit.ViTCoreMLConfig"
        ),
        "yolos": supported_features_mapping(
            "default",
            "object-detection",
            coreml_config_cls="models.yolos.YolosCoreMLConfig",
        ),
    }

    AVAILABLE_FEATURES = sorted(reduce(lambda s1, s2: s1 | s2, (v.keys() for v in _SUPPORTED_MODEL_TYPE.values())))

    @staticmethod
    def get_supported_features_for_model_type(
        model_type: str, model_name: Optional[str] = None
    ) -> Dict[str, Callable[[PretrainedConfig], CoreMLConfig]]:
        """
        Tries to retrieve the feature -> CoreMLConfig constructor map from the model type.

        Args:
            model_type (`str`):
                The model type to retrieve the supported features for.
            model_name (`str`, *optional*):
                The name attribute of the model object, only used for the exception message.

        Returns:
            The dictionary mapping each feature to a corresponding CoreMLConfig constructor.
        """
        model_type = model_type.lower()
        if model_type not in FeaturesManager._SUPPORTED_MODEL_TYPE:
            model_type_and_model_name = f"{model_type} ({model_name})" if model_name else model_type
            raise KeyError(
                f"{model_type_and_model_name} is not supported yet. "
                f"Only {list(FeaturesManager._SUPPORTED_MODEL_TYPE.keys())} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type]

    @staticmethod
    def feature_to_task(feature: str) -> str:
        return feature.replace("-with-past", "")

    @staticmethod
    def _validate_framework_choice(framework: str):
        """
        Validates if the framework requested for the export is both correct and available, otherwise throws an
        exception.
        """
        if framework not in ["pt", "tf"]:
            raise ValueError(
                f"Only two frameworks are supported for Core ML export: pt or tf, but {framework} was provided."
            )
        elif framework == "pt" and not is_torch_available():
            raise RuntimeError("Cannot export model to Core ML using PyTorch because no PyTorch package was found.")
        elif framework == "tf" and not is_tf_available():
            raise RuntimeError("Cannot export model to Core ML using TensorFlow because no TensorFlow package was found.")

    @staticmethod
    def get_model_class_for_feature(feature: str, framework: str = "pt") -> Type:
        """
        Attempts to retrieve an AutoModel class from a feature name.

        Args:
            feature (`str`):
                The feature required.
            framework (`str`, *optional*, defaults to `"pt"`):
                The framework to use for the export.

        Returns:
            The AutoModel class corresponding to the feature.
        """
        task = FeaturesManager.feature_to_task(feature)
        FeaturesManager._validate_framework_choice(framework)
        if framework == "pt":
            task_to_automodel = FeaturesManager._TASKS_TO_AUTOMODELS
        else:
            task_to_automodel = FeaturesManager._TASKS_TO_TF_AUTOMODELS
        if task not in task_to_automodel:
            raise KeyError(
                f"Unknown task: {feature}. Possible values are {list(FeaturesManager._TASKS_TO_AUTOMODELS.values())}"
            )
        return task_to_automodel[task]

    @staticmethod
    def get_model_from_feature(
        feature: str, model: str, framework: str = "pt", cache_dir: str = None
    ) -> Union["PreTrainedModel", "TFPreTrainedModel"]:
        """
        Attempts to retrieve a model from a model's name and the feature to be enabled.

        Args:
            feature (`str`):
                The feature required.
            model (`str`):
                The name of the model to export.
            framework (`str`, *optional*, defaults to `"pt"`):
                The framework to use for the export.

        Returns:
            The instance of the model.

        """
        model_class = FeaturesManager.get_model_class_for_feature(feature, framework)
        try:
            model = model_class.from_pretrained(model, cache_dir=cache_dir, torchscript=True)
        except OSError:
            if framework == "pt":
                model = model_class.from_pretrained(model, from_tf=True, cache_dir=cache_dir)
            else:
                model = model_class.from_pretrained(model, from_pt=True, cache_dir=cache_dir, torchscript=True)
        return model

    @staticmethod
    def check_supported_model_or_raise(
        model: Union["PreTrainedModel", "TFPreTrainedModel"], feature: str = "default"
    ) -> Tuple[str, Callable]:
        """
        Check whether or not the model has the requested features.

        Args:
            model: The model to export.
            feature: The name of the feature to check if it is available.

        Returns:
            (str) The type of the model (CoreMLConfig) The CoreMLConfig instance holding the model export properties.

        """
        model_type = model.config.model_type.replace("_", "-")
        model_name = getattr(model, "name", "")
        model_features = FeaturesManager.get_supported_features_for_model_type(model_type, model_name=model_name)
        if feature not in model_features:
            raise ValueError(
                f"{model.config.model_type} doesn't support feature {feature}. Supported values are: {model_features}"
            )

        return model.config.model_type, FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]

    @staticmethod
    def get_config(model_type: str, feature: str) -> CoreMLConfig:
        """
        Gets the `CoreMLConfig` for a model_type and feature combination.

        Args:
            model_type (`str`):
                The model type to retrieve the config for.
            feature (`str`):
                The feature to retrieve the config for.

        Returns:
            `CoreMLConfig`: config for the combination
        """
        return FeaturesManager._SUPPORTED_MODEL_TYPE[model_type][feature]
