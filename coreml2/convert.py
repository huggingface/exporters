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

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

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
            the Core ML model object
        output (`ct.proto.Model_pb2.FeatureDescription`):
            the output protobuf object from the Core ML model's spec
        name (`str`):
            the new name for the model output
        description (`str`):
            the new description for the model output
        shape (`tuple`, *optional*, default is `None`):
            the expected shape for the output tensor
    """
    ct.utils.rename_feature(mlmodel._spec, output.name, name)
    mlmodel.output_description[name] = description
    if shape is not None:
        set_multiarray_shape(output, shape)


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

    Returns:
        `ct.models.MLModel`: the Core ML model object
    """
    if not issubclass(type(model), PreTrainedModel):
        raise ValueError(f"Cannot convert unknown model type: {type(model)}")

    logger.info(f"Using framework PyTorch: {torch.__version__}")

    # TODO: get the scale + bias from the Config object

    feature_extractor = preprocessor

    scale = 1.0 / (feature_extractor.image_std[0] * 255)
    bias = [
        -feature_extractor.image_mean[0] / feature_extractor.image_std[0],
        -feature_extractor.image_mean[1] / feature_extractor.image_std[1],
        -feature_extractor.image_mean[2] / feature_extractor.image_std[2],
    ]

    # TODO: get dummy input from the Config object
    # NOTE: we're going to be applying the std in the Wrapper so do that thing
    # where we use a different std on the feature extractor?
    #    maybe that isn't needed for the dummy inputs

    image_size = feature_extractor.size
    image_shape = (1, 3, image_size, image_size)
    pixel_values = torch.rand(image_shape) * 2.0 - 1.0
    example_input = [ pixel_values ]

    # TODO: also add to Config dummy inputs

    if config.task == "masked-im":
        num_patches = (model.config.image_size // model.config.patch_size) ** 2
        bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
        example_input.append(bool_masked_pos)

    wrapper = Wrapper(preprocessor, model, config).eval()
    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    # Run the PyTorch model, to get the shapes of the output tensors.
    with torch.no_grad():
        example_output = traced_model(*example_input)

    if isinstance(example_output, (tuple, list)):
        example_output = [x.numpy() for x in example_output]
    else:
        example_output = example_output.numpy()

    convert_kwargs = { }
    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    # if isinstance(torch_model, ViTForImageClassification):
    if config.task == "image-classification":
        class_labels = [model.config.id2label[x] for x in range(model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    # pass any additional arguments to ct.convert()
    # for key, value in kwargs.items():
    #     convert_kwargs[key] = value

    # TODO: depends on task / Config
    input_tensors = [ ct.ImageType(name="image", shape=image_shape, scale=scale, bias=bias,
                                   color_layout="RGB", channel_first=True) ]

    if config.task == "masked-im":
        input_tensors.append(ct.TensorType(name="bool_masked_pos", shape=bool_masked_pos.shape, dtype=np.int32))

    mlmodel = ct.convert(
        traced_model,
        inputs=input_tensors,
        convert_to="neuralnetwork" if legacy else "mlprogram",
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if model.config.transformers_version:
        user_defined_metadata["transformers_version"] = model.config.transformers_version

    #TODO: only if image model
    mlmodel.input_description["image"] = "Image input"

    if config.task == "image-classification":
        probs_output_name = spec.description.predictedProbabilitiesName
        ct.utils.rename_feature(spec, probs_output_name, "probabilities")
        spec.description.predictedProbabilitiesName = "probabilities"

        mlmodel.input_description["image"] = "Image to be classified"
        mlmodel.output_description["probabilities"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if isinstance(example_output, (tuple, list)):
        first_output_shape = example_output[0].shape
    else:
        first_output_shape = example_output.shape

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

    Returns:
        `ct.models.MLModel`: the Core ML model object
    """
    if not (is_torch_available() or is_tf_available()):
        raise ImportError(
            "Cannot convert because neither PyTorch nor TensorFlow are not installed. "
            "Please install torch or tensorflow first."
        )

    if is_torch_available() and issubclass(type(model), PreTrainedModel):
        return export_pytorch(preprocessor, model, config, quantize, legacy)
    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        return export_tensorflow(preprocessor, model, config, quantize, legacy)
    else:
        raise ValueError(f"Cannot convert unknown model type: {type(model)}")
