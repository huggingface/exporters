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

import dataclasses
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np

from transformers.utils import (
    TensorType,
    is_torch_available,
    is_vision_available,
    logging,
)


if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


if is_vision_available():
    from PIL import Image


logger = logging.get_logger(__name__)


@dataclasses.dataclass
class InputDescription:
    """
    Data class that describes the properties of a Core ML model input.

    Args:
        name (`str`):
            Name of the input in the Core ML model. This does not have to be the same as the
            input name in the original Transformers model.
        description (`str`, *optional*, defaults to empty string):
            Input description in the Core ML Model.
        is_optional (`bool`, *optional*, defaults to `False`):
            If true, this input may be omitted.
        sequence_length (`int` or tuple, *optional*, defaults to `None`):
            Sequence length for text inputs. If this is a single value, the sequence length will be a fixed size
            in the exported model, giving the input tensor the shape `(batch_size, sequence_length)`.
            If this is a tuple `(min, max)`, the sequence length is allowed to vary between those two sizes.
        color_layout (`str`, *optional*, defaults to `None`):
            Channel ordering for image inputs. Either `"RGB"` or `"BGR"`.
    """
    name: str
    description: str = ""
    is_optional: bool = False
    sequence_length: Optional[Union[int, Tuple[int, int]]] = None
    color_layout: Optional[str] = None


@dataclasses.dataclass
class OutputDescription:
    """
    Data class that describes the properties of a Core ML model output.

    Args:
        name (`str`):
            Name of the output in the Core ML model. This does not have to be the same as the name
            in the output dictionary of the original Transformers model.
        description (`str`, *optional*, defaults to empty string):
            Output description in the Core ML Model.
        do_softmax (`bool`, *optional*, defaults to `None`):
            For tasks that output logits: Applies a softmax to the logits.
        do_upsample (`bool`, *optional*, defaults to `None`):
            For the `"semantic-segmentation"` task: Resizes the output to have the same width and height as the input.
        do_argmax (`bool`, *optional*, defaults to `None`):
            For the `"semantic-segmentation"` task: Whether to perform an argmax operation on the predicted logits.
    """
    name: str
    description: str = ""
    do_softmax: Optional[bool] = None
    do_upsample: Optional[bool] = None
    do_argmax: Optional[bool] = None


class CoreMLConfig():
    """
    Base class for Core ML exportable model describing metadata on how to export the model through the Core ML format.
    """
    def __init__(
        self,
        config: "PretrainedConfig",
        task: str,
        modality: str,
        use_past: bool = False,
        seq2seq: bool = False,     # TODO: get from model type
    ):
        self._config = config
        self.task = task
        self.modality = modality
        self.use_past = use_past
        self.seq2seq = seq2seq

    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "default") -> "CoreMLConfig":
        """
        Instantiate a `CoreMLConfig` with `use_past` attribute set to True

        Args:
            config: The model's configuration to use when exporting to Core ML.
            task: The model topology that will be exported.

        Returns:
            `CoreMLVisionConfig` for this model with `.use_past = True`
        """
        return cls(config, task=task, use_past=True)

    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        """
        Ordered mapping of the inputs from the original model to the exported Core ML model.
        """
        common_inputs = self._input_descriptions

        if self.use_past:
            self.fill_inputs_with_past_key_values_(common_inputs)

        return common_inputs

    @property
    def _input_descriptions(self) -> OrderedDict[str, InputDescription]:
        if self.modality == "text" and self.task in [
            "default",
            "causal-lm",
            "masked-lm",
            "question-answering",
            "sequence-classification",
            "token-classification",
        ]:
            return OrderedDict(
                [
                    (
                        "input_ids",
                        InputDescription(
                            "input_ids",
                            "Indices of input sequence tokens in the vocabulary",
                            sequence_length=(1, 128),
                        )
                    ),
                    (
                        "attention_mask",
                        InputDescription(
                            "attention_mask",
                            "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        )
                    ),
                ]
            )

        if self.task in [
            "multiple-choice",
            "next-sentence-prediction",
        ]:
            return OrderedDict(
                [
                    (
                        "input_ids",
                        InputDescription(
                            "input_ids",
                            "Indices of input sequence tokens in the vocabulary",
                            sequence_length=(1, 128),
                        )
                    ),
                    (
                        "attention_mask",
                        InputDescription(
                            "attention_mask",
                            "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        )
                    ),
                    (
                        "token_type_ids",
                        InputDescription(
                            "token_type_ids",
                            "Segment token indices to indicate first and second portions of the inputs (0 = sentence A, 1 = sentence B)",
                        )
                    ),
                ]
            )

        if self.modality == "vision" and self.task in [
            "default",
            "object-detection",
            "semantic-segmentation"
        ]:
            return OrderedDict(
                [
                    (
                        "pixel_values",
                        InputDescription("image", "Input image", color_layout="RGB")
                    ),
                ]
            )

        if self.task == "image-classification":
            return OrderedDict(
                [
                    (
                        "pixel_values",
                        InputDescription("image", "Image to be classified", color_layout="RGB")
                    ),
                ]
            )

        if self.task == "masked-im":
            return OrderedDict(
                [
                    (
                        "pixel_values",
                        InputDescription("image", "Image to be classified", color_layout="RGB")
                    ),
                    (
                        "bool_masked_pos",
                        InputDescription("bool_masked_pos", "Indicates which patches are masked (1) and which aren't (0)"),
                    ),
                ]
            )

        raise AssertionError("Unsupported task '{self.task}' for modality `{self.modality}`")

    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        """
        Ordered mapping of the outputs from the original model to the exported Core ML model.
        """
        common_outputs = self._output_descriptions

        if self.use_past:
            self.fill_outputs_with_past_key_values_(common_outputs)

        return common_outputs

    @property
    def _output_descriptions(self) -> OrderedDict[str, OutputDescription]:
        if self.task == "default":
            return OrderedDict(
                [
                    (
                        "last_hidden_state",
                        OutputDescription(
                            "last_hidden_state",
                            "Sequence of hidden-states at the output of the last layer of the model",
                        )
                    ),
                ]
            )

        if self.task in [
            "image-classification",
            "multiple-choice",
            "next-sentence-prediction",
            "sequence-classification",
        ]:
            return OrderedDict(
                [
                    (
                        "logits",
                        OutputDescription(
                            "probabilities",
                            "Probability of each category",
                            do_softmax=True,
                        )
                    ),
                    (
                        "class_labels",
                        OutputDescription(
                            "classLabel",
                            "Category with the highest score",
                        )
                    ),
                ]
            )

        if self.task == "masked-im":
            return OrderedDict(
                [
                    (
                        "logits",
                        OutputDescription(
                            "logits",
                            "Classification scores (before softmax)",
                            do_softmax=False,
                        )
                    ),
                ]
            )

        if self.task in ["causal-lm", "masked-lm", "token-classification"]:
            return OrderedDict(
                [
                    (
                        "logits",
                        OutputDescription(
                            "token_scores",
                            "Classification scores for each vocabulary token (after softmax)",
                            do_softmax=True,
                        )
                    ),
                ]
            )

        if self.task == "object-detection":
            return OrderedDict(
                [
                    (
                        "logits",
                        OutputDescription(
                            "logits",
                            "Classification logits (including no-object) for all queries",
                            do_softmax=False,
                        )
                    ),
                    (
                        "pred_boxes",
                        OutputDescription(
                            "pred_boxes",
                            "Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height)",
                        )
                    ),
                ]
            )

        if self.task == "question-answering":
            return OrderedDict(
                [
                    (
                        "start_logits",
                        OutputDescription(
                            "start_scores",
                            "Span-start scores (after softmax)",
                            do_softmax=True,
                        )
                    ),
                    (
                        "end_logits",
                        OutputDescription(
                            "end_scores",
                            "Span-end scores (after softmax)",
                            do_softmax=True,
                        )
                    ),
                ]
            )

        if self.task == "semantic-segmentation":
            return OrderedDict(
                [
                    (
                        "logits",
                        OutputDescription(
                            "classLabels",
                            "Classification scores for each pixel",
                            do_softmax=False,
                            do_upsample=True,
                            do_argmax=True,
                        )
                    ),
                ]
            )

        raise AssertionError("Unsupported task '{self.task}' for modality `{self.modality}`")

    def get_flexible_outputs(self) -> Mapping[str, List[Mapping[str, int]]]:
        """
        Determines which outputs require flexible shapes and on which axes.

        Flexible output shapes are used when `sequence_length` on the model input is a range of
        allowed lengths.
        """
        output_shapes = {}

        # Only tasks that output a sequence need a flexible output shape.
        if self.task in ["default", "causal-lm", "masked-lm", "question-answering", "token-classification"]:
            input_descs = self.inputs
            output_descs = self.outputs

            # If this model has flexible input shapes, it also needs flexible output shapes.
            if self.use_past:
                min_length, max_length = 1, -1
            elif "input_ids" in input_descs and isinstance(input_descs["input_ids"].sequence_length, tuple):
                min_length, max_length = input_descs["input_ids"].sequence_length
            else:
                min_length, max_length = None, None

            if min_length is not None:
                for key in ["last_hidden_state", "logits", "start_logits", "end_logits"]:
                    if key in output_descs:
                        output_shapes[key] = [{ "axis": 1, "min": min_length, "max": max_length }]

        if self.use_past:
            for i in range(self.num_layers):
                output_shapes[f"present_{i}_key"] = [{ "axis": 2, "min": 1, "max": -1 }]
                output_shapes[f"present_{i}_value"] = [{ "axis": 2, "min": 1, "max": -1 }]

        return output_shapes

    @property
    def num_layers(self) -> int:
        """
        The number of layers retrieved from the model config. Override this for model configs where the
        number of layers attribute is not called `num_layers`.
        """
        if hasattr(self._config, "num_hidden_layers"):
            return self._config.num_hidden_layers

        if hasattr(self._config, "n_layer"):
            return self._config.n_layer

        if not hasattr(self._config, "num_layers"):
            raise AttributeError(
                "could not find the number of layers attribute in the model configuration, override the num_layers"
                " property of the model CoreMLConfig to solve this"
            )
        return self._config.num_layers

    @property
    def num_attention_heads(self) -> int:
        """
        The number of attention heads retrieved from the model config. Override this for model configs where
        the number of attention heads attribute is not called `num_attention_heads`.
        """
        if not hasattr(self._config, "num_attention_heads"):
            raise AttributeError(
                "could not find the number of attention heads attribute in the model configuration, override the"
                " num_attention_heads property of the model CoreMLConfig to solve this"
            )
        return self._config.num_attention_heads

    def fill_inputs_with_past_key_values_(self, inputs: OrderedDict[str, InputDescription]):
        name = "past_key_values"
        for i in range(self.num_layers):
            inputs[f"{name}_{i}_key"] = InputDescription(f"{name}_{i}_key", is_optional=True)
            inputs[f"{name}_{i}_value"] = InputDescription(f"{name}_{i}_value", is_optional=True)

    def fill_outputs_with_past_key_values_(self, outputs: OrderedDict[str, OutputDescription]):
        name = "present"
        for i in range(self.num_layers):
            outputs[f"{name}_{i}_key"] = OutputDescription(f"{name}_{i}_key")
            outputs[f"{name}_{i}_value"] = OutputDescription(f"{name}_{i}_value")

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting.

        Returns:
            Dictionary with the keys (and their corresponding values) to override.
        """
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}

        return None

    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        return 1e-4

    @property
    def use_legacy_format(self) -> bool:
        """
        If `True`, the converter will produce a model in the older NeuralNetwork format.
        By default, the ML Program format will be used.
        """
        return False

    def patch_pytorch_ops(self) -> Mapping[str, Callable]:
        """
        Override this to provide implementation for PyTorch ops that the Core ML
        converter does not support.

        Returns:
            `Mapping[str, Callable]` of op names to PyTorch conversion functions.
        """
        return {}

    @property
    def is_classifier(self) -> bool:
        """
        Determines whether this is a treated as a special classifier model by Core ML.
        """
        classifier_tasks = [
            "image-classification",
            "multiple-choice",
            "next-sentence-prediction",
            "sequence-classification"
        ]
        return self.task in classifier_tasks and self.outputs["logits"].do_softmax

    def _rename_duplicate_labels(self, labels):
        """
        Renames duplicate label names. Core ML puts the labels as keys into a dictionary,
        and so all the label names need to be unique.
        """
        unique_labels = []
        used_labels = set()

        for label in labels:
            while label in used_labels:
                label = label + "_duplicate"

            used_labels.add(label)
            unique_labels.append(label)

        if len(unique_labels) != len(set(unique_labels)):
            raise AssertionError("Unable to remove duplicates from the provided labels")

        return unique_labels

    def get_class_labels(self) -> List[str]:
        """
        Return the model's classification labels as a sorted list.
        """
        labels = [self._config.id2label[x] for x in range(self._config.num_labels)]

        if len(labels) != len(set(labels)):
            labels = self._rename_duplicate_labels(labels)

        return labels

    def _generate_dummy_image(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        framework: Optional[TensorType] = None,
    ) -> Tuple[Any, Any]:
        if hasattr(preprocessor, "crop_size") and preprocessor.do_center_crop:
            image_size = preprocessor.crop_size
        else:
            image_size = preprocessor.size

        if isinstance(image_size, tuple):
            image_width, image_height = image_size
        else:
            image_width = image_height = image_size

        pixel_values = np.random.randint(0, 256, (image_width, image_height, 3), dtype=np.uint8)
        coreml_value = Image.fromarray(pixel_values)

        # Hacky workaround: the Core ML input is the full-sized image, and so
        # the feature extractor should not resize or crop it, only normalize.
        old_do_resize = None
        if hasattr(preprocessor, "do_resize"):
            old_do_resize = preprocessor.do_resize
            preprocessor.do_resize = False

        old_crop_pct = None
        if hasattr(preprocessor, "crop_pct"):
            old_crop_pct = preprocessor.crop_pct
            preprocessor.crop_pct = 1.0

        ref_value = preprocessor(coreml_value, return_tensors=framework)["pixel_values"]

        if old_do_resize is not None:
            preprocessor.do_resize = old_do_resize
        if old_crop_pct is not None:
            preprocessor.crop_pct = old_crop_pct

        return (ref_value, coreml_value)

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Tuple[Any, Any]]:
        """
        Generate dummy input data to provide to the Core ML exporter.

        Args:
            preprocessor: ([`PreTrainedTokenizerBase`] or [`FeatureExtractionMixin`]):
                The preprocessor associated with this model configuration.
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.

        Returns:
            `Mapping[str, Tuple[Any, Any]]` holding tuples containing the reference and
            Core ML tensors to provide to the model's forward function.
        """
        from transformers.feature_extraction_utils import FeatureExtractionMixin
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        input_descs = self.inputs
        dummy_inputs = {}

        if self.modality == "text" and isinstance(preprocessor, PreTrainedTokenizerBase):
            input_desc = input_descs["input_ids"]

            # the dummy input will always use the maximum sequence length
            if input_desc.sequence_length is None:
                sequence_length = 64
            elif isinstance(input_desc.sequence_length, tuple):
                sequence_length = input_desc.sequence_length[-1]
            else:
                sequence_length = input_desc.sequence_length

            if self.task == "multiple-choice":
                shape = (1, self._config.num_labels, sequence_length)
            else:
                shape = (1, sequence_length)

            input_ids = np.random.randint(0, preprocessor.vocab_size, shape)
            dummy_inputs["input_ids"] = (input_ids, input_ids.astype(np.int32))

            if "attention_mask" in input_descs:
                attention_mask = np.ones(shape, dtype=np.int64)
                dummy_inputs["attention_mask"] = (attention_mask, attention_mask.astype(np.int32))

            if "token_type_ids" in input_descs:
                token_type_ids = np.zeros(shape, dtype=np.int64)
                dummy_inputs["token_type_ids"] = (token_type_ids, token_type_ids.astype(np.int32))

        elif (
            self.modality == "vision"
            and isinstance(preprocessor, FeatureExtractionMixin)
            and preprocessor.model_input_names[0] == "pixel_values"
        ):
            dummy_inputs["pixel_values"] = self._generate_dummy_image(preprocessor, framework)

            if self.task == "masked-im":
                num_patches = (self._config.image_size // self._config.patch_size) ** 2
                bool_masked_pos = np.random.randint(low=0, high=2, size=(1, num_patches)).astype(bool)
                dummy_inputs["bool_masked_pos"] = (bool_masked_pos, bool_masked_pos.astype(np.int32))

        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )

        if self.use_past:
            batch, seqlen = dummy_inputs["input_ids"][0].shape

            # Not using the same length for past_key_values
            past_key_values_length = seqlen + 2
            shape = (
                batch,
                self.num_attention_heads,
                past_key_values_length,
                self._config.hidden_size // self.num_attention_heads,
            )

            if "attention_mask" in dummy_inputs:
                attention_mask = np.ones((batch, seqlen + past_key_values_length), dtype=np.int64)
                dummy_inputs["attention_mask"] = (attention_mask, attention_mask.astype(np.int32))

            for i in range(self.num_layers):
                dummy_inputs[f"past_key_values_{i}_key"] = (
                    np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
                )
                dummy_inputs[f"past_key_values_{i}_value"] = (
                    np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
                )

        return self._convert_dummy_inputs_to_framework(dummy_inputs, framework)

    def _convert_dummy_inputs_to_framework(self, dummy_inputs, framework):
        if framework == TensorType.PYTORCH and is_torch_available():
            import torch
            for key, (ref_value, coreml_value) in dummy_inputs.items():
                if isinstance(ref_value, np.ndarray):
                    dummy_inputs[key] = (torch.tensor(ref_value), coreml_value)

        return dummy_inputs

    def _add_pooler_output(self, output_descs):
        if self.task == "default":
            if self.modality == "vision":
                description = "Last layer hidden-state after a pooling operation on the spatial dimensions"
            else:
                description = "Last layer hidden-state of the first token of the sequence"

            output_descs["pooler_output"] = OutputDescription(
                "pooler_output",
                description
            )
        return output_descs


class CoreMLTextConfig(CoreMLConfig):
    """
    Base class for Core ML exportable model using the "text" modality, describing metadata on how to
    export the model through the Core ML format.
    """
    def __init__(self, config: "PretrainedConfig", task: str = "default", use_past: bool = False):
        super().__init__(config, task=task, modality="text", use_past=use_past)

    @classmethod
    def from_model_config(cls, config: "PretrainedConfig", task: str = "default", use_past: bool = False) -> "CoreMLTextConfig":
        """
        Instantiate a `CoreMLConfig` for a specific model.

        Args:
            config: The model's configuration to use when exporting to Core ML.
            task: The model topology that will be exported.

        Returns:
            `CoreMLTextConfig` for this model
        """
        return cls(config, task=task, use_past=use_past)


class CoreMLVisionConfig(CoreMLConfig):
    """
    Base class for Core ML exportable model using the "vision" modality, describing metadata on how to
    export the model through the Core ML format.
    """
    def __init__(self, config: "PretrainedConfig", task: str = "default"):
        super().__init__(config, task=task, modality="vision")

    @classmethod
    def from_model_config(cls, config: "PretrainedConfig", task: str = "default") -> "CoreMLVisionConfig":
        """
        Instantiate a `CoreMLConfig` for a specific model.

        Args:
            config: The model's configuration to use when exporting to Core ML.
            task: The model topology that will be exported.

        Returns:
            `CoreMLVisionConfig` for this model
        """
        return cls(config, task=task)
