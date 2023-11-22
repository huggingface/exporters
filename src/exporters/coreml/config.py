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
)
from ..utils import logging


if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.image_processing_utils import ImageProcessingMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.processing_utils import ProcessorMixin


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

    Args:
        config: The model's configuration to use when exporting to Core ML.
        task: The model topology that will be exported.
        use_past: Export the model with precomputed hidden states (key and values in the
            attention blocks) for fast autoregressive decoding.
        seq2seq: `None` if not an encoder-decoder model, `"encoder"` to export the encoder
            part of a seq2seq model, `"decoder"` to export the decoder part.
    """
    def __init__(
        self,
        config: "PretrainedConfig",
        task: str,
        use_past: bool = False,
        seq2seq: Optional[str] = None,
    ):
        if not hasattr(self, "modality"):
            raise ValueError("the CoreMLConfig subclass must have a modality property")

        if use_past and seq2seq == "encoder":
            raise ValueError("invalid option `use_past=True` for encoder model")

        self._config = config
        self.task = task
        self.use_past = use_past
        self.seq2seq = seq2seq

    @classmethod
    def from_model_config(
        cls,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        use_past: bool = False,
        seq2seq: Optional[str] = None,
    ) -> "CoreMLConfig":
        """
        Instantiate a `CoreMLConfig` for a specific model.

        Args:
            config: The model's configuration to use when exporting to Core ML.
            task: The model topology that will be exported.
            use_past: Export the model with precomputed hidden states (key and values in the
                attention blocks) for fast autoregressive decoding.
            seq2seq: `None` if not an encoder-decoder model, `"encoder"` to export the encoder
                part of a seq2seq model, `"decoder"` to export the decoder part.

        Returns:
            `CoreMLConfig` for this model
        """
        return cls(config, task=task, use_past=use_past, seq2seq=seq2seq)

    @classmethod
    def with_past(
        cls,
        config: "PretrainedConfig",
        task: str = "feature-extraction",
        seq2seq: Optional[str] = None,
    ) -> "CoreMLConfig":
        """
        Instantiate a `CoreMLConfig` with `use_past` attribute set to True

        Args:
            config: The model's configuration to use when exporting to Core ML.
            task: The model topology that will be exported.
            seq2seq: `None` if not an encoder-decoder model, `"encoder"` to export the encoder
                part of a seq2seq model, `"decoder"` to export the decoder part.

        Returns:
            `CoreMLVisionConfig` for this model with `.use_past = True`
        """
        return cls(config, task=task, use_past=True, seq2seq=seq2seq)

    @property
    def inputs(self) -> "OrderedDict[str, InputDescription]":
        """
        Ordered mapping of the inputs from the original model to the exported Core ML model.
        """
        common_inputs = self._input_descriptions

        if self.use_past:
            self.fill_inputs_with_past_key_values_(common_inputs)

        return common_inputs

    @property
    def infer_sequence_length_from_config(self) -> bool:
        """When True, will use the max sequence length from the model configuration."""
        return False

    @property
    def max_sequence_length(self) -> int:
        """
        Retrieve the max sequence length from the model configuration, or use a hardcoded value (currently 128).
        This can be subclassed to support custom lengths.
        """
        if self.infer_sequence_length_from_config:
            # Alternatives such as `n_positions` are automatically mapped to `max_position_embeddings`
            if hasattr(self._config, "max_position_embeddings"):
                return self._config.max_position_embeddings
        return 128

    @property
    def use_flexible_shapes(self) -> bool:
        """
        When True, inputs are allowed to use sequence lengths of `1` up to `maxSequenceLength`.
        Unfortunately, this currently prevents the model from running on GPU or the Neural Engine.
        We default to `False`, but this can be overridden in custom configurations.
        """
        return False

    @property
    def input_ids_sequence_length(self) -> Union[Tuple, int]:
        """
        Sequence lengths supported for the `input_ids`.

        - When returning a tuple, flexible shapes will be used. The tuple must contain two items,
        representing the minimum and maximum possible sequence lengths.
        - When returning an `int`, a fixed sequence length will be used.
        """
        return (1, self.max_sequence_length) if self.use_flexible_shapes else self.max_sequence_length


    @property
    def _input_descriptions(self) -> "OrderedDict[str, InputDescription]":
        if self.modality in ["text", "audio"] and self.seq2seq == "decoder":
            return OrderedDict(
                [
                    (
                        "decoder_input_ids",
                        InputDescription(
                            "decoder_input_ids",
                            "Indices of decoder input sequence tokens in the vocabulary",
                        )
                    ),
                    (
                        "decoder_attention_mask",
                        InputDescription(
                            "decoder_attention_mask",
                            "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        )
                    ),
                    (
                        "encoder_outputs",
                        InputDescription(
                            "encoder_last_hidden_state",
                            "Sequence of hidden states at the output of the last layer of the encoder",
                        )
                    ),
                    (
                        "attention_mask",
                        InputDescription(
                            "encoder_attention_mask",
                            "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        )
                    ),
                ]
            )

        if self.modality == "text" and self.task in [
            "feature-extraction",
            "text-generation",
            "fill-mask",
            "question-answering",
            "text-classification",
            "text2text-generation",
            "token-classification",
        ]:
            return OrderedDict(
                [
                    (
                        "input_ids",
                        InputDescription(
                            "input_ids",
                            "Indices of input sequence tokens in the vocabulary",
                            sequence_length=self.input_ids_sequence_length,
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
                            sequence_length=self.input_ids_sequence_length,
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
            "feature-extraction",
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

        if self.modality == "audio":
            if self._get_mel_bins() > 0:
                audio_input = (
                    "input_features",
                    InputDescription(
                        "input_features",
                        "Mel features extracted from the raw speech waveform",
                        sequence_length=(1, -1),
                    )
                )
            else:
                audio_input =  (
                    "input_values",
                    InputDescription(
                        "input_values",
                        "Raw speech waveform",
                        sequence_length=(1, -1),
                    )
                )

            return OrderedDict(
                [
                    audio_input,
                    (
                        "attention_mask",
                        InputDescription(
                            "attention_mask",
                            "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)",
                        )
                    ),
                ]
            )

        raise AssertionError(f"Unsupported task '{self.task}' for modality '{self.modality}'")

    @property
    def outputs(self) -> "OrderedDict[str, OutputDescription]":
        """
        Ordered mapping of the outputs from the original model to the exported Core ML model.
        """
        common_outputs = self._output_descriptions

        if self.use_past:
            self.fill_outputs_with_past_key_values_(common_outputs)

        return common_outputs

    @property
    def _output_descriptions(self) -> "OrderedDict[str, OutputDescription]":
        if self.task == "feature-extraction" or self.seq2seq == "encoder":
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
            "text-classification",
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

        if self.task in [
            "masked-im",
            "text-generation",
            "text2text-generation",
        ]:
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

        if self.task in [
            "automatic-speech-recognition",
            "fill-mask",
            "speech-seq2seq",
            "token-classification"
        ]:
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

        raise AssertionError(f"Unsupported task '{self.task}' for modality '{self.modality}'")

    def get_flexible_outputs(self) -> Mapping[str, List[Mapping[str, int]]]:
        """
        Determines which outputs require flexible shapes and on which axes.

        Flexible output shapes are used when `sequence_length` on the model input is a range of
        allowed lengths.
        """
        output_shapes = {}

        # Only tasks that output a sequence need a flexible output shape.
        if self.task in [
            "feature-extraction",
            "text-generation",
            "automatic-speech-recognition",
            "fill-mask",
            "question-answering",
            "text2text-generation",
            "speech-seq2seq",
            "token-classification",
        ]:
            input_descs = self.inputs
            output_descs = self.outputs

            # If this model has flexible input shapes, it also needs flexible output shapes.
            min_length, max_length = None, None
            if self.use_past or self.seq2seq:
                min_length, max_length = 1, -1
            else:
                sequence_length = self.get_input_sequence_length(input_descs)
                if isinstance(sequence_length, tuple):
                    min_length, max_length = sequence_length

            if min_length is not None:
                for key in ["last_hidden_state", "logits", "start_logits", "end_logits"]:
                    if key in output_descs:
                        output_shapes[key] = [
                            #{ "axis": 0, "min": 1, "max": -1 },  # batch size  # TODO
                            { "axis": 1, "min": min_length, "max": max_length },
                        ]

        if self.use_past:
            # TODO: Temporarily disabled until we can solve the issue with encoder past key/values
            #name = "decoder_present" if self.seq2seq == "decoder" else "present"
            name = "present"
            for i in range(self.num_layers):
                output_shapes[f"{name}_{i}_key"] = [
                    #{ "axis": 0, "min": 1, "max": -1 },  # batch size  # TODO
                    { "axis": 2, "min": 1, "max": -1 },
                ]
                output_shapes[f"{name}_{i}_value"] = [
                    #{ "axis": 0, "min": 1, "max": -1 },  # batch size  # TODO
                    { "axis": 2, "min": 1, "max": -1 },
                ]

            # TODO: Temporarily disabled until we can solve the issue with encoder past key/values
            # if self.seq2seq == "decoder":
            #     for i in range(self.num_encoder_layers):
            #         output_shapes[f"encoder_present_{i}_key"] = [
            #             { "axis": 2, "min": 1, "max": -1 },
            #         ]
            #         output_shapes[f"encoder_present_{i}_value"] = [
            #             { "axis": 2, "min": 1, "max": -1 },
            #         ]

        return output_shapes

    def get_input_sequence_length(self, input_descs):
        if "input_ids" in input_descs:
            return input_descs["input_ids"].sequence_length
        elif "input_values" in input_descs:
            return input_descs["input_values"].sequence_length
        elif "input_features" in input_descs:
            return input_descs["input_features"].sequence_length
        else:
            return None

    @property
    def num_layers(self) -> int:
        """
        The number of layers retrieved from the model config. Override this for model configs where the
        number of layers attribute is not called `num_layers`.
        """
        if self.seq2seq == "encoder" and hasattr(self._config, "encoder_layers"):
            return self._config.encoder_layers

        if self.seq2seq == "decoder" and hasattr(self._config, "decoder_layers"):
            return self._config.decoder_layers

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
    def num_encoder_layers(self) -> int:
        """
        The number of encoder layers retrieved from the model config of an encoder-decoder model.
        """
        return getattr(self._config, "encoder_layers", self.num_layers)

    @property
    def num_attention_heads(self) -> int:
        """
        The number of attention heads retrieved from the model config. Override this for model configs where
        the number of attention heads attribute is not called `num_attention_heads`.
        """
        if self.seq2seq == "encoder" and hasattr(self._config, "encoder_attention_heads"):
            return self._config.encoder_attention_heads

        if self.seq2seq == "decoder" and hasattr(self._config, "decoder_attention_heads"):
            return self._config.decoder_attention_heads

        if not hasattr(self._config, "num_attention_heads"):
            raise AttributeError(
                "could not find the number of attention heads attribute in the model configuration, override the"
                " num_attention_heads property of the model CoreMLConfig to solve this"
            )
        return self._config.num_attention_heads

    def fill_inputs_with_past_key_values_(self, inputs: "OrderedDict[str, InputDescription]"):
        # TODO: Temporarily disabled until we can solve the issue with encoder past key/values
        #name = "decoder_past_key_values" if self.seq2seq == "decoder" else "past_key_values"
        name = "past_key_values"
        for i in range(self.num_layers):
            inputs[f"{name}_{i}_key"] = InputDescription(f"{name}_{i}_key", is_optional=True)
            inputs[f"{name}_{i}_value"] = InputDescription(f"{name}_{i}_value", is_optional=True)

        # TODO: Temporarily disabled until we can solve the issue with encoder past key/values
        # if self.seq2seq == "decoder":
        #     name = "encoder_past_key_values"
        #     for i in range(self.num_encoder_layers):
        #         inputs[f"{name}_{i}_key"] = InputDescription(f"{name}_{i}_key", is_optional=True)
        #         inputs[f"{name}_{i}_value"] = InputDescription(f"{name}_{i}_value", is_optional=True)

    def fill_outputs_with_past_key_values_(self, outputs: "OrderedDict[str, OutputDescription]"):
        # TODO: Temporarily disabled until we can solve the issue with encoder past key/values
        # name = "decoder_present" if self.seq2seq == "decoder" else "present"
        name = "present"
        for i in range(self.num_layers):
            outputs[f"{name}_{i}_key"] = OutputDescription(f"{name}_{i}_key")
            outputs[f"{name}_{i}_value"] = OutputDescription(f"{name}_{i}_value")

        # TODO: Temporarily disabled until we can solve the issue with encoder past key/values
        # if self.seq2seq == "decoder":
        #     name = "encoder_present"
        #     for i in range(self.num_encoder_layers):
        #         outputs[f"{name}_{i}_key"] = OutputDescription(f"{name}_{i}_key")
        #         outputs[f"{name}_{i}_value"] = OutputDescription(f"{name}_{i}_value")

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
            "text-classification"
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
        preprocessor: "ImageProcessingMixin",
        framework: Optional[TensorType] = None,
    ) -> Tuple[Any, Any]:
        if hasattr(preprocessor, "crop_size") and preprocessor.do_center_crop:
            image_size = preprocessor.crop_size
        else:
            image_size = preprocessor.size

        if "shortest_edge" in image_size:
            image_height = image_width = image_size["shortest_edge"]
        else:
            image_height = image_size["height"]
            image_width = image_size["width"]

        pixel_values = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)
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

    def _get_max_sequence_length(self, input_desc, default_length):
        if input_desc.sequence_length is None:
            return self.max_sequence_length
        elif isinstance(input_desc.sequence_length, tuple):
            sequence_length = input_desc.sequence_length[-1]
            if sequence_length == -1:
                sequence_length = default_length
            return sequence_length
        else:
            return input_desc.sequence_length

    def _get_mel_bins(self):
        if hasattr(self._config, "num_mel_bins"):
            return self._config.num_mel_bins
        elif hasattr(self._config, "input_feat_per_channel"):
            return self._config.input_feat_per_channel
        else:
            return 0

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "ImageProcessingMixin", "ProcessorMixin"],
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Tuple[Any, Any]]:
        """
        Generate dummy input data to provide to the Core ML exporter.

        Args:
            preprocessor: ([`PreTrainedTokenizerBase`] or [`ImageProcessingMixin`] or [`ProcessorMixin`]):
                The preprocessor associated with this model configuration.
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the preprocessor will generate tensors for.

        Returns:
            `Mapping[str, Tuple[Any, Any]]` holding tuples containing the reference and
            Core ML tensors to provide to the model's forward function.
        """
        from transformers.image_processing_utils import ImageProcessingMixin
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        from transformers.processing_utils import ProcessorMixin

        batch_size = 1
        input_descs = self.inputs
        dummy_inputs = {}

        if self.modality == "text" and isinstance(preprocessor, PreTrainedTokenizerBase):
            if self.seq2seq == "decoder":
                input_ids_name = "decoder_input_ids"
                attention_mask_name = "decoder_attention_mask"
            else:
                input_ids_name = "input_ids"
                attention_mask_name = "attention_mask"

            input_desc = input_descs[input_ids_name]

            # the dummy input will always use the maximum sequence length
            sequence_length = self._get_max_sequence_length(input_desc, 64)

            # don't want encoder and decoder to use same sequence length
            # (unless shapes are fixed)
            if self.seq2seq == "decoder":
                if isinstance(input_desc.sequence_length, tuple):
                    encoder_sequence_length = sequence_length + 7
                else:
                    encoder_sequence_length = sequence_length

            if self.task == "multiple-choice":
                shape = (batch_size, self._config.num_labels, sequence_length)
            else:
                shape = (batch_size, sequence_length)

            input_ids = np.random.randint(0, preprocessor.vocab_size, shape)
            dummy_inputs[input_ids_name] = (input_ids, input_ids.astype(np.int32))

            if attention_mask_name in input_descs:
                attention_mask = np.ones(shape, dtype=np.int64)
                dummy_inputs[attention_mask_name] = (attention_mask, attention_mask.astype(np.int32))

            if "token_type_ids" in input_descs:
                token_type_ids = np.zeros(shape, dtype=np.int64)
                dummy_inputs["token_type_ids"] = (token_type_ids, token_type_ids.astype(np.int32))

            if "encoder_outputs" in input_descs:
                last_hidden_state = np.zeros((batch_size, encoder_sequence_length, self._config.hidden_size), dtype=np.float32)
                dummy_inputs["encoder_outputs"] = (last_hidden_state, last_hidden_state)

            if self.seq2seq == "decoder" and "attention_mask" in input_descs:
                encoder_attention_mask = np.ones((batch_size, encoder_sequence_length), dtype=np.int64)
                dummy_inputs["attention_mask"] = (encoder_attention_mask, encoder_attention_mask.astype(np.int32))

            if self.task == "feature-extraction" and "decoder_input_ids" in input_descs:
                # Special case for T5-like models
                decoder_shape = (batch_size, sequence_length-5)
                decoder_input_ids = np.random.randint(0, preprocessor.vocab_size, decoder_shape)
                dummy_inputs["decoder_input_ids"] = (decoder_input_ids, decoder_input_ids.astype(np.int32))

                decoder_attention_mask = np.ones(decoder_shape, dtype=np.int64)
                dummy_inputs["decoder_attention_mask"] = (decoder_attention_mask, decoder_attention_mask.astype(np.int32))

        elif (
            self.modality == "vision"
            and isinstance(preprocessor, ImageProcessingMixin)
            and preprocessor.model_input_names[0] == "pixel_values"
        ):
            dummy_inputs["pixel_values"] = self._generate_dummy_image(preprocessor, framework)

            if self.task == "masked-im":
                num_patches = (self._config.image_size // self._config.patch_size) ** 2
                bool_masked_pos = np.random.randint(low=0, high=2, size=(1, num_patches)).astype(bool)
                dummy_inputs["bool_masked_pos"] = (bool_masked_pos, bool_masked_pos.astype(np.int32))

        elif self.modality == "audio" and isinstance(preprocessor, ProcessorMixin):
            if self.seq2seq != "decoder":
                if "input_features" in input_descs:
                    mel_bins = self._get_mel_bins()
                    if mel_bins == 0:
                        raise ValueError("Cannot determine number of mel bins from model config")

                    # TODO: some models (e.g. Whisper) may put the mel bins on another axis

                    input_desc = input_descs["input_features"]  # mel filterbanks
                    sequence_length = self._get_max_sequence_length(input_desc, 200)
                    input_features = np.random.rand(batch_size, sequence_length, mel_bins).astype(np.float32)
                    dummy_inputs["input_features"] = (input_features, input_features)
                else:
                    input_desc = input_descs["input_values"]  # raw audio
                    sequence_length = self._get_max_sequence_length(input_desc, 50000)
                    input_features = np.random.rand(batch_size, sequence_length).astype(np.float32) * 2.0 - 1.0
                    dummy_inputs["input_values"] = (input_features, input_features)

                if "attention_mask" in input_descs:
                    attention_mask = np.ones((batch_size, sequence_length), dtype=np.int64)
                    dummy_inputs["attention_mask"] = (attention_mask, attention_mask.astype(np.int32))

            else:  # decoder
                input_desc = input_descs["decoder_input_ids"]
                sequence_length = 64
                shape = (batch_size, sequence_length)

                input_ids = np.random.randint(0, preprocessor.tokenizer.vocab_size, shape)
                dummy_inputs["decoder_input_ids"] = (input_ids, input_ids.astype(np.int32))

                if "decoder_attention_mask" in input_descs:
                    attention_mask = np.ones(shape, dtype=np.int64)
                    dummy_inputs["decoder_attention_mask"] = (attention_mask, attention_mask.astype(np.int32))

                if "encoder_outputs" in input_descs:
                    last_hidden_state = np.zeros((batch_size, self._config.max_source_positions, self._config.hidden_size), dtype=np.float32)
                    dummy_inputs["encoder_outputs"] = (last_hidden_state, last_hidden_state)

                if "attention_mask" in input_descs:
                    encoder_attention_mask = np.ones((batch_size, self._config.max_source_positions), dtype=np.int64)
                    dummy_inputs["attention_mask"] = (encoder_attention_mask, encoder_attention_mask.astype(np.int32))

        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )

        if self.use_past:
            batch, sequence_length = dummy_inputs[input_ids_name][0].shape

            # Not using the same length for past_key_values
            past_key_values_length = sequence_length + 2
            shape = (
                batch,
                self.num_attention_heads,
                past_key_values_length,
                self._config.hidden_size // self.num_attention_heads,
            )

            # Resize the attention mask to include the past
            if attention_mask_name in dummy_inputs:
                attention_mask = np.ones((batch, sequence_length + past_key_values_length), dtype=np.int64)
                dummy_inputs[attention_mask_name] = (attention_mask, attention_mask.astype(np.int32))

            # TODO: Temporarily disabled until we can solve the issue with encoder past key/values
            # name = "decoder_past_key_values" if self.seq2seq == "decoder" else "past_key_values"
            name = "past_key_values"
            for i in range(self.num_layers):
                dummy_inputs[f"{name}_{i}_key"] = (
                    np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
                )
                dummy_inputs[f"{name}_{i}_value"] = (
                    np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
                )

            # TODO: Temporarily disabled until we can solve the issue with encoder past key/values
            # Encoder-decoder model also needs past_key_values for encoder sequence
            # if self.seq2seq == "decoder":
            #     encoder_sequence_length = sequence_length + 7
            #     shape = (
            #         batch,
            #         self.num_attention_heads,
            #         encoder_sequence_length,
            #         self._config.hidden_size // self.num_attention_heads,
            #     )
            #     name = "encoder_past_key_values"
            #     for i in range(self.num_encoder_layers):
            #         dummy_inputs[f"{name}_{i}_key"] = (
            #             np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
            #         )
            #         dummy_inputs[f"{name}_{i}_value"] = (
            #             np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
            #         )

        return self._convert_dummy_inputs_to_framework(dummy_inputs, framework)

    def _convert_dummy_inputs_to_framework(self, dummy_inputs, framework):
        if framework == TensorType.PYTORCH and is_torch_available():
            import torch
            for key, (ref_value, coreml_value) in dummy_inputs.items():
                if isinstance(ref_value, np.ndarray):
                    dummy_inputs[key] = (torch.tensor(ref_value), coreml_value)

        return dummy_inputs

    def _add_pooler_output(self, output_descs):
        if self.task == "feature-extraction":
            if self.modality == "vision":
                description = "Last layer hidden-state after a pooling operation on the spatial dimensions"
            else:
                description = "Last layer hidden-state of the first token of the sequence"

            output_descs["pooler_output"] = OutputDescription(
                "pooler_output",
                description
            )
        return output_descs
    
    @property
    def short_description(self) -> str:
        """
        Short description: name and task.
        """
        return f"{self._config.name_or_path} ({self.task})"
