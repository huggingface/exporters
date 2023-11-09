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

from transformers import PretrainedConfig, is_tf_available, is_torch_available
from .config import CoreMLConfig
from ..utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TFPreTrainedModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_available():
    from transformers.models.auto import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForCTC,
        AutoModelForImageClassification,
        # AutoModelForImageSegmentation,
        AutoModelForMaskedImageModeling,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForNextSentencePrediction,
        AutoModelForObjectDetection,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSemanticSegmentation,
        AutoModelForSequenceClassification,
        AutoModelForSpeechSeq2Seq,
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

    import exporters.coreml.models
    config_cls = exporters.coreml
    for attr_name in coreml_config_cls.split("."):
        if not hasattr(config_cls, attr_name): continue  # hack!
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
            "feature-extraction": AutoModel,
            "text-generation": AutoModelForCausalLM,
            "automatic-speech-recognition": AutoModelForCTC,
            "image-classification": AutoModelForImageClassification,
            # "image-segmentation": AutoModelForImageSegmentation,
            "masked-im": AutoModelForMaskedImageModeling,
            "fill-mask": AutoModelForMaskedLM,
            "multiple-choice": AutoModelForMultipleChoice,
            "next-sentence-prediction": AutoModelForNextSentencePrediction,
            "object-detection": AutoModelForObjectDetection,
            "question-answering": AutoModelForQuestionAnswering,
            "semantic-segmentation": AutoModelForSemanticSegmentation,
            "text2text-generation": AutoModelForSeq2SeqLM,
            "text-classification": AutoModelForSequenceClassification,
            "speech-seq2seq": AutoModelForSpeechSeq2Seq,
            "token-classification": AutoModelForTokenClassification,
        }
    if is_tf_available():
        _TASKS_TO_TF_AUTOMODELS = {
            # "feature-extraction": TFAutoModel,
            # "text-generation": TFAutoModelForCausalLM,
            # "fill-mask": TFAutoModelForMaskedLM,
            # "multiple-choice": TFAutoModelForMultipleChoice,
            # "question-answering": TFAutoModelForQuestionAnswering,
            # "text2text-generation": TFAutoModelForSeq2SeqLM,
            # "text-classification": TFAutoModelForSequenceClassification,
            # "token-classification": TFAutoModelForTokenClassification,
        }

    _SYNONYM_TASK_MAP = {
        "sequence-classification": "text-classification",
        "causal-lm": "text-generation",
        "causal-lm-with-past": "text-generation-with-past",
        "seq2seq-lm": "text2text-generation",
        "seq2seq-lm-with-past": "text2text-generation-with-past",
        "speech2seq-lm": "automatic-speech-recognition",
        "speech2seq-lm-with-past": "automatic-speech-recognition-with-past",
        "masked-lm": "fill-mask",
        "vision2seq-lm": "image-to-text",
        "default": "feature-extraction",
        "default-with-past": "feature-extraction-with-past",
        "automatic-speech-recognition": "automatic-speech-recognition",
        "ctc": "automatic-speech-recognition",
    }

    _SUPPORTED_MODEL_TYPE = {
        "bart": supported_features_mapping(
            "feature-extraction",
            "text-generation",
            "text2text-generation",
            coreml_config_cls="models.bart.BartCoreMLConfig",
        ),
        # BEiT cannot be used with the masked image modeling autoclass, so this feature is excluded here
        "beit": supported_features_mapping(
            "feature-extraction",
            "image-classification",
            "semantic-segmentation",
            coreml_config_cls="models.beit.BeitCoreMLConfig"
        ),
        "bert": supported_features_mapping(
            "feature-extraction",
            "fill-mask",
            "text-generation",
            "text-generation-with-past",
            "multiple-choice",
            "next-sentence-prediction",
            "question-answering",
            "text-classification",
            "token-classification",
            coreml_config_cls="models.bert.BertCoreMLConfig",
        ),
        "big_bird": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.big_bird.BigBirdCoreMLConfig",
        ),
        "bigbird_pegasus": supported_features_mapping(
            "feature-extraction",
            "text-generation",
            "text-generation-with-past",
            "text2text-generation",
            coreml_config_cls="models.bigbird_pegasus.BigBirdPegasusCoreMLConfig",
        ),
        "blenderbot": supported_features_mapping(
            "feature-extraction",
            "text2text-generation",
            coreml_config_cls="models.blenderbot.BlenderbotCoreMLConfig",
        ),
        "blenderbot_small": supported_features_mapping(
            "feature-extraction",
            "text2text-generation",
            coreml_config_cls="models.blenderbot_small.BlenderbotSmallCoreMLConfig",
        ),
        "bloom": supported_features_mapping(
            "feature-extraction",
            "text-generation",
            coreml_config_cls="models.bloom.BloomCoreMLConfig",
        ),
        "convnext": supported_features_mapping(
            "feature-extraction",
            "image-classification",
            coreml_config_cls="models.convnext.ConvNextCoreMLConfig",
        ),
        "ctrl": supported_features_mapping(
            "feature-extraction",
            "feature-extraction-with-past",
            "text-generation",
            "text-generation-with-past",
            "text-classification",
            coreml_config_cls="models.ctrl.CTRLCoreMLConfig",
        ),
        "cvt": supported_features_mapping(
            "feature-extraction",
            "image-classification",
            coreml_config_cls="models.cvt.CvtCoreMLConfig",
        ),
        "data2vec": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.data2vec.Data2VecTextCoreMLConfig",
        ),
        "distilbert": supported_features_mapping(
            "feature-extraction",
            "fill-mask",
            "multiple-choice",
            "question-answering",
            "text-classification",
            "token-classification",
            coreml_config_cls="models.distilbert.DistilBertCoreMLConfig",
        ),
        "ernie": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.ernie.ErnieCoreMLConfig",
        ),
        "falcon": supported_features_mapping(
            "feature-extraction",
            "text-generation",
            "text-classification",
            coreml_config_cls="models.falcon.FalconCoreMLConfig",
        ),
        "gpt2": supported_features_mapping(
            "feature-extraction",
            #"feature-extraction-with-past",
            "text-generation",
            #"text-generation-with-past",
            "text-classification",
            "token-classification",
            coreml_config_cls="models.gpt2.GPT2CoreMLConfig",
        ),
        "gpt_bigcode": supported_features_mapping(
            "feature-extraction",
            "text-generation",
            "text-classification",
            coreml_config_cls="models.gpt_bigcode.GPTBigcodeCoreMLConfig",
        ),
        "gptj": supported_features_mapping(
            "feature-extraction",
            "text-generation",
            coreml_config_cls="models.gpt2.GPTJCoreMLConfig",
        ),
        "gpt_neo": supported_features_mapping(
            "feature-extraction",
            #"feature-extraction-with-past",
            "text-generation",
            #"text-generation-with-past",
            "text-classification",
            coreml_config_cls="models.gpt_neo.GPTNeoCoreMLConfig",
        ),
        "gpt_neox": supported_features_mapping(
            "feature-extraction",
            #"feature-extraction-with-past",
            "text-generation",
            #"text-generation-with-past",
            "text-classification",
            coreml_config_cls="models.gpt_neox.GPTNeoXCoreMLConfig",
        ),
        "levit": supported_features_mapping(
            "feature-extraction", "image-classification", coreml_config_cls="models.levit.LevitCoreMLConfig"
        ),
        "llama": supported_features_mapping(
            "feature-extraction",
            "text-generation",
            "text-classification",
            coreml_config_cls="models.llama.LlamaCoreMLConfig",
        ),
        "m2m_100": supported_features_mapping(
            "feature-extraction",
            "text2text-generation",
            coreml_config_cls="models.m2m_100.M2M100CoreMLConfig",
        ),
        "marian": supported_features_mapping(
            "feature-extraction",
            "text2text-generation",
            coreml_config_cls="models.marian.MarianMTCoreMLConfig",
        ),
        "mistral": supported_features_mapping(
            "feature-extraction",
            "text-generation",
            "text-classification",
            coreml_config_cls="models.mistral.MistralCoreMLConfig",
        ),
        "mobilebert": supported_features_mapping(
            "feature-extraction",
            "fill-mask",
            "multiple-choice",
            "next-sentence-prediction",
            "question-answering",
            "text-classification",
            "token-classification",
            coreml_config_cls="models.mobilebert.MobileBertCoreMLConfig",
        ),
        "mobilevit": supported_features_mapping(
            "feature-extraction",
            "image-classification",
            "semantic-segmentation",
            coreml_config_cls="models.mobilevit.MobileViTCoreMLConfig",
        ),
        "mobilevitv2": supported_features_mapping(
            "feature-extraction",
            "image-classification",
            "semantic-segmentation",
            coreml_config_cls="models.mobilevit.MobileViTCoreMLConfig",
        ),
        "mvp": supported_features_mapping(
            "feature-extraction",
            "text2text-generation",
            coreml_config_cls="models.mvp.MvpCoreMLConfig",
        ),
        "pegasus": supported_features_mapping(
            "feature-extraction",
            "text2text-generation",
            coreml_config_cls="models.pegasus.PegasusCoreMLConfig",
        ),
        "plbart": supported_features_mapping(
            "feature-extraction",
            "text2text-generation",
            coreml_config_cls="models.plbart.PLBartCoreMLConfig",
        ),
        "roberta": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.roberta.RobertaCoreMLConfig",
        ),
        "roformer": supported_features_mapping(
            "text-generation",
            "text-generation-with-past",
            coreml_config_cls="models.roformer.RoFormerCoreMLConfig",
        ),
        "segformer": supported_features_mapping(
            "feature-extraction",
            "image-classification",
            "semantic-segmentation",
            coreml_config_cls="models.segformer.SegformerCoreMLConfig",
        ),
        "splinter": supported_features_mapping(
            "feature-extraction",
            "text-generation-with-past",
            coreml_config_cls="models.splinter.SplinterCoreMLConfig",
        ),
        "squeezebert": supported_features_mapping(
            "feature-extraction",
            "fill-mask",
            "multiple-choice",
            "question-answering",
            "text-classification",
            "token-classification",
            coreml_config_cls="models.squeezebert.SqueezeBertCoreMLConfig",
        ),
        "t5": supported_features_mapping(
            "feature-extraction",
            "text2text-generation",
            coreml_config_cls="models.t5.T5CoreMLConfig",
        ),
        "vit": supported_features_mapping(
            "feature-extraction", "image-classification", "masked-im", coreml_config_cls="models.vit.ViTCoreMLConfig"
        ),
        "yolos": supported_features_mapping(
            "feature-extraction",
            "object-detection",
            coreml_config_cls="models.yolos.YolosCoreMLConfig",
        ),
    }

    AVAILABLE_FEATURES = sorted(reduce(lambda s1, s2: s1 | s2, (v.keys() for v in _SUPPORTED_MODEL_TYPE.values())))
    AVAILABLE_FEATURES_INCLUDING_LEGACY = AVAILABLE_FEATURES + list(_SYNONYM_TASK_MAP.keys())

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
    def map_from_synonym(feature: str) -> str:
        if feature in FeaturesManager._SYNONYM_TASK_MAP:
            feature = FeaturesManager._SYNONYM_TASK_MAP[feature]
        return feature

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
        model: Union["PreTrainedModel", "TFPreTrainedModel"], feature: str = "feature-extraction"
    ) -> Tuple[str, Callable]:
        """
        Check whether or not the model has the requested features.

        Args:
            model: The model to export.
            feature: The name of the feature to check if it is available.

        Returns:
            (str) The type of the model (CoreMLConfig) The CoreMLConfig instance holding the model export properties.

        """
        model_type = model.config.model_type.replace("-", "_")
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
