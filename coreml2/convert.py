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

from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Mapping, Any

import coremltools as ct
import numpy as np

#TODO: if integrating this into transformers, replace imports with ..

from transformers.utils import (
    is_torch_available,
    is_tf_available,
    logging,
)
from .config import CoreMLConfig


if is_torch_available():
    from transformers.modeling_utils import PreTrainedModel

if is_tf_available():
    from transformers.modeling_tf_utils import TFPreTrainedModel

if TYPE_CHECKING:
    from transformers.feature_extraction_utils import FeatureExtractionMixin
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# def get_input_named(spec, name):
#     """Return the input node with the given name in the Core ML model."""
#     for out in spec.description.input:
#         if out.name == name:
#             return out
#     return None


def get_output_names(spec):
    """Return a list of all output names in the Core ML model."""
    outputs = []
    for out in spec.description.output:
        outputs.append(out.name)
    return outputs


def get_output_named(spec, name):
    """Return the output node with the given name in the Core ML model."""
    for out in spec.description.output:
        if out.name == name:
            return out
    return None


def set_multiarray_shape(node, shape):
    """Change the shape of the specified input or output in the Core ML model."""
    del node.type.multiArrayType.shape[:]
    for x in shape:
        node.type.multiArrayType.shape.append(x)


def get_labels_as_list(model):
    """Return the labels of a classifier model as a sorted list."""
    labels = []
    for i in range(len(model.config.id2label)):
        if i in model.config.id2label.keys():
            labels.append(model.config.id2label[i])
    return labels


def fix_output(
    mlmodel: ct.models.MLModel,
    output: ct.proto.Model_pb2.FeatureDescription,
    name: str,
    description: str,
    shape: Optional[Tuple[int]] = None
):
    """
    Rename model output and fill in its expected shape

    Args:
        mlmodel (`ct.models.MLModel`):
            The Core ML model object.
        output (`ct.proto.Model_pb2.FeatureDescription`):
            The output protobuf object from the Core ML model's spec.
        name (`str`):
            The new name for the model output.
        description (`str`):
            The new description for the model output.
        shape (`tuple`, *optional*, default is `None`):
            The expected shape for the output tensor.
    """
    ct.utils.rename_feature(mlmodel._spec, output.name, name)
    mlmodel.output_description[name] = description
    if shape is not None:
        set_multiarray_shape(output, shape)


def _get_input_types(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    config: CoreMLConfig,
    dummy_inputs: Mapping[str, np.ndarray],
) -> List[Union[ct.ImageType, ct.TensorType]]:
    """
    Create the ct.InputType objects that describe the inputs to the Core ML model

    Args:
        preprocessor ([`PreTrainedTokenizer`], [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            The preprocessor used for encoding the data.
        config ([`~coreml.config.CoreMLConfig`]):
            The Core ML configuration associated with the exported model.
        dummy_inputs (`Mapping[str, np.ndarray]`):
            The dummy input tensors that describe the expected shapes of the inputs.

    Returns:
        `List[Union[ct.ImageType, ct.TensorType]]`: ordered list of input types
    """
    input_defs = config.inputs
    input_types = []

    if config.task in ["image-classification", "masked-im"]:
        scale = 1.0 / (preprocessor.image_std[0] * 255.0)
        bias = [
            -preprocessor.image_mean[0] / preprocessor.image_std[0],
            -preprocessor.image_mean[1] / preprocessor.image_std[1],
            -preprocessor.image_mean[2] / preprocessor.image_std[2],
        ]

        input_name, input_config = input_defs.popitem(last=False)
        color_layout = input_config.get("color_layout", "RGB")

        input_types.append(
            ct.ImageType(
                name=input_name,
                shape=dummy_inputs[input_name].shape,
                scale=scale,
                bias=bias,
                color_layout=color_layout,
                channel_first=True,
            )
        )

    if config.task == "masked-im":
        input_name, input_config = input_defs.popitem(last=False)

        input_types.append(
            ct.TensorType(
                name=input_name,
                shape=dummy_inputs[input_name].shape,
                dtype=np.int32,
            )
        )

    return input_types


if is_torch_available():
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, preprocessor, model, config):
            super().__init__()
            self.preprocessor = preprocessor
            self.model = model.eval()
            self.config = config

        def forward(self, inputs, extra_input1=None):
            if self.config.task == "masked-im":
                outputs = self.model(inputs, bool_masked_pos=extra_input1, return_dict=False)
            else:
                outputs = self.model(inputs, return_dict=False)

            if self.config.task == "image-classification":
                return torch.nn.functional.softmax(outputs[0], dim=1)  # logits

            if self.config.task == "masked-im":
                return outputs[1]  # logits

            if self.config.task == "default":
                if hasattr(self.model, "pooler") and self.model.pooler is not None:
                    return outputs[0], outputs[1]  # last_hidden_state, pooler_output
                else:
                    return outputs[0]  # last_hidden_state

            raise AssertionError(f"Cannot compute outputs for unknown task '{self.config.task}'")


def export_pytorch(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    model: "PreTrainedModel",
    config: CoreMLConfig,
    quantize: str = "float32",
    legacy: bool = False,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> ct.models.MLModel:
    """
    Export a PyTorch model to Core ML format

    Args:
        preprocessor ([`PreTrainedTokenizer`], [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            The preprocessor used for encoding the data.
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~coreml.config.CoreMLConfig`]):
            The Core ML configuration associated with the exported model.
        quantize (`str`, *optional*, defaults to `"float32"`):
            Quantization options. Possible values: `"float32"`, `"float16"`.
        legacy (`bool`, *optional*, defaults to `False`):
            If `True`, the converter will produce a model in the older NeuralNetwork format.
            By default, the ML Program format will be used.
        compute_units (`ct.ComputeUnit`, *optional*, defaults to `ct.ComputeUnit.ALL`):
            Whether to optimize the model for CPU, GPU, and/or Neural Engine.

    Returns:
        `ct.models.MLModel`: the Core ML model object
    """
    if not issubclass(type(model), PreTrainedModel):
        raise ValueError(f"Cannot convert unknown model type: {type(model)}")

    logger.info(f"Using framework PyTorch: {torch.__version__}")

    # Create dummy input data for doing the JIT trace.
    dummy_inputs = config.generate_dummy_inputs(preprocessor)

    # Convert to Torch tensors and use inputs in order from the config.
    example_input = [torch.tensor(dummy_inputs[name]) for name in list(config.inputs.keys())]

    wrapper = Wrapper(preprocessor, model, config).eval()
    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    # Run the traced PyTorch model to get the shapes of the output tensors.
    with torch.no_grad():
        example_output = traced_model(*example_input)

    if isinstance(example_output, (tuple, list)):
        example_output = [x.numpy() for x in example_output]
    else:
        example_output = example_output.numpy()

    convert_kwargs = { }

    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    # For classification models, add the labels into the Core ML model and
    # designate it as the special `classifier` model type.
    if config.task == "image-classification":
        class_labels = [model.config.id2label[x] for x in range(model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    input_tensors = _get_input_types(preprocessor, config, dummy_inputs)

    mlmodel = ct.convert(
        traced_model,
        inputs=input_tensors,
        convert_to="neuralnetwork" if legacy else "mlprogram",
        compute_units=compute_units,
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if model.config.transformers_version:
        user_defined_metadata["transformers_version"] = model.config.transformers_version

    if isinstance(example_output, (tuple, list)):
        first_output_shape = example_output[0].shape
    else:
        first_output_shape = example_output.shape

    if config.task == "image-classification":
        probs_output_name = spec.description.predictedProbabilitiesName
        ct.utils.rename_feature(spec, probs_output_name, "probabilities")
        spec.description.predictedProbabilitiesName = "probabilities"

        mlmodel.output_description["probabilities"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if config.task == "masked-im":
        fix_output(
            mlmodel=mlmodel,
            output=spec.description.output[0],
            name="logits",
            description="Prediction scores (before softmax)",
            shape=first_output_shape,
        )

    if config.task == "default":
        fix_output(
            mlmodel=mlmodel,
            output=spec.description.output[0],
            name="last_hidden_state",
            description="Hidden states from the last layer",
            shape=first_output_shape,
        )

        if hasattr(model, "pooler") and model.pooler is not None:
            fix_output(
                mlmodel=mlmodel,
                output=spec.description.output[1],
                name="pooler_output",
                description="Output from the global pooling layer",
                shape=example_output[1].shape,
            )

    #TODO: no longer pass description to fix_output but get it from config.outputs

    for input_name, input_config in config.inputs.items():
        if "description" in input_config:
            mlmodel.input_description[input_name] = input_config["description"]

    if len(user_defined_metadata) > 0:
        spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    if legacy and quantize == "float16":
        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=16)

    return mlmodel


def export_tensorflow(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin"],
    model: "TFPreTrainedModel",
    config: CoreMLConfig,
    quantize: str = "float32",
    legacy: bool = False,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> ct.models.MLModel:
    """
    Export a TensorFlow model to Core ML format

    Args:
        preprocessor ([`PreTrainedTokenizer`] or [`FeatureExtractionMixin`]):
            The preprocessor used for encoding the data.
        model ([`TFPreTrainedModel`]):
            The model to export.
        config ([`~coreml.config.CoreMLConfig`]):
            The Core ML configuration associated with the exported model.
        quantize (`str`, *optional*, defaults to `"float32"`):
            Quantization options. Possible values: `"float32"`, `"float16"`.
        legacy (`bool`, *optional*, defaults to `False`):
            If `True`, the converter will produce a model in the older NeuralNetwork format.
            By default, the ML Program format will be used.
        compute_units (`ct.ComputeUnit`, *optional*, defaults to `ct.ComputeUnit.ALL`):
            Whether to optimize the model for CPU, GPU, and/or Neural Engine.

    Returns:
        `ct.models.MLModel`: the Core ML model object
    """
    raise AssertionError(f"Core ML export does not currently support TensorFlow models")


def export(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    config: CoreMLConfig,
    quantize: str = "float32",
    legacy: bool = False,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
) -> ct.models.MLModel:
    """
    Export a Pytorch or TensorFlow model to Core ML format

    Args:
        preprocessor ([`PreTrainedTokenizer`], [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            The preprocessor used for encoding the data.
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~coreml.config.CoreMLConfig`]):
            The Core ML configuration associated with the exported model.
        quantize (`str`, *optional*, defaults to `"float32"`):
            Quantization options. Possible values: `"float32"`, `"float16"`.
        legacy (`bool`, *optional*, defaults to `False`):
            If `True`, the converter will produce a model in the older NeuralNetwork format.
            By default, the ML Program format will be used.
        compute_units (`ct.ComputeUnit`, *optional*, defaults to `ct.ComputeUnit.ALL`):
            Whether to optimize the model for CPU, GPU, and/or Neural Engine.

    Returns:
        `ct.models.MLModel`: the Core ML model object
    """
    if not (is_torch_available() or is_tf_available()):
        raise ImportError(
            "Cannot convert because neither PyTorch nor TensorFlow are not installed. "
            "Please install torch or tensorflow first."
        )

    if is_torch_available() and issubclass(type(model), PreTrainedModel):
        return export_pytorch(preprocessor, model, config, quantize, legacy, compute_units)
    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        return export_tensorflow(preprocessor, model, config, quantize, legacy, compute_units)
    else:
        raise ValueError(f"Cannot convert unknown model type: {type(model)}")
