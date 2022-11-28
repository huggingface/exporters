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

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple, Union

import coremltools as ct
import numpy as np

from transformers.utils import TensorType, is_torch_available
from transformers.modeling_utils import PreTrainedModel

from .config import CoreMLConfig
from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def softmax(x, axis=-1):
    maxes = np.max(x, axis=axis, keepdims=True)
    shifted_exp = np.exp(x - maxes)
    return shifted_exp / shifted_exp.sum(axis=axis, keepdims=True)


def validate_model_outputs(
    config: CoreMLConfig,
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    reference_model: Union["PreTrainedModel", "TFPreTrainedModel"],
    mlmodel: ct.models.MLModel,
    atol: float,
):
    """
    Validate that the outputs from the base and exported model agree within some absolute tolerance.

    Args:
        config ([`~coreml.config.CoreMLConfig`]):
            The Core ML configuration associated with the exported model.
        preprocessor ([`PreTrainedTokenizer`], [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            The preprocessor used for encoding the data.
        reference_model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        mlmodel (`ct.models.MLModel`):
            The exported Core ML model.
        atol (`float`):
            Absolute tolerance. Differences larger than this value are considered problematic.
    """
    logger.info("Validating Core ML model...")

    input_descs = config.inputs
    output_descs = config.outputs

    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        framework = TensorType.PYTORCH
    else:
        framework = TensorType.TENSORFLOW

    dummy_inputs = config.generate_dummy_inputs(preprocessor, framework)

    reference_model_inputs = {}
    past_key_values = []
    coreml_inputs = {}

    # Put the dummy inputs into Core ML and reference model input dictionaries.
    # The separate past_key_values inputs are combined into a tuple of tuples.
    for name in input_descs.keys():
        ref_value, coreml_value = dummy_inputs[name]
        if name.startswith("past_key_values_"):
            if name.endswith("_key"):
                past_key_values.append((ref_value,))
            else:
                past_key_values[-1] += (ref_value,)
        elif name == "encoder_outputs":
            reference_model_inputs[name] = (ref_value,)
        else:
            reference_model_inputs[name] = ref_value
        coreml_inputs[input_descs[name].name] = coreml_value

    if len(past_key_values) > 0:
        reference_model_inputs["past_key_values"] = past_key_values

    # Compute outputs from the reference model
    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        reference_model.to("cpu").eval()
    if config.seq2seq == "encoder":
        reference_model = reference_model.get_encoder()
    ref_outputs_dict = reference_model(**reference_model_inputs, return_dict=True)

    # Unpack the past_key_values output into separate outputs, as that is also
    # how the Core ML mdel does it.
    if "past_key_values" in ref_outputs_dict:
        for i in range(len(ref_outputs_dict["past_key_values"])):
            ref_outputs_dict[f"present_{i}_key"] = ref_outputs_dict["past_key_values"][i][0]
            ref_outputs_dict[f"present_{i}_value"] = ref_outputs_dict["past_key_values"][i][1]

    # Compute outputs from the Core ML model
    coreml_outputs = mlmodel.predict(coreml_inputs)

    # Map the Core ML output names back to the names used by the reference model
    coreml_output_names = list(coreml_outputs.keys())
    coreml_output_internal_names = []
    for name, desc in output_descs.items():
        if desc.name in coreml_output_names:
            coreml_output_internal_names.append(name)

    spec = mlmodel._spec

    # Classifier models are special in Core ML
    if config.is_classifier:
        logger.info("\t- Core ML model is classifier, validating output")

        if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
            ref_logits = ref_outputs_dict["logits"].detach().numpy()
        else:
            ref_logits = ref_outputs_dict["logits"].numpy()

        labels_name = spec.description.predictedFeatureName
        coreml_value = coreml_outputs[labels_name]

        ref_value = reference_model.config.id2label[np.argmax(ref_logits, axis=-1)[0]]
        if coreml_value != ref_value:
            logger.info(f"\t\t-[x] predicted class '{coreml_value}' doesn't match '{ref_value}'")
            raise ValueError(
                "Predicted class doesn't match between reference model and Core ML exported model: "
                f"Got {ref_value} (reference) and {coreml_value} (Core ML)"
            )
        else:
            logger.info(f"\t\t-[✓] predicted class '{coreml_value}' matches '{ref_value}'")

        probs_name = spec.description.predictedProbabilitiesName
        coreml_value = coreml_outputs[probs_name]
        ref_value = softmax(ref_logits, axis=-1)[0]

        # Shape
        if len(coreml_value) != len(ref_value):
            logger.info(f"\t\t-[x] number of classes {len(coreml_value)} doesn't match {len(ref_value)}")
            raise ValueError(
                "Output shape doesn't match between reference model and Core ML exported model: "
                f"Got {len(ref_value)} (reference) and {len(coreml_value)} (Core ML)"
            )
        else:
            logger.info(f"\t\t-[✓] number of classes {len(coreml_value)} matches {len(ref_value)}")

        # Core ML probabilities are in a dict, put in sorted list for comparing
        class_labels = config.get_class_labels()
        coreml_probs = np.zeros_like(ref_value)
        for i in range(len(ref_value)):
            coreml_probs[i] = coreml_value[class_labels[i]]

        # Values
        if not np.allclose(ref_value, coreml_probs, atol=atol):
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Output values do not match between reference model and Core ML exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - coreml_probs))}"
            )
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")

        return

    # Check that keys in coreml_output_internal are a subset of keys from ref_outputs
    ref_outputs_set = set(ref_outputs_dict.keys())
    coreml_outputs_set = set(coreml_output_internal_names)
    if not coreml_outputs_set.issubset(ref_outputs_set):
        logger.info(
            f"\t-[x] Core ML model output names {coreml_outputs_set} do not match reference model {ref_outputs_set}"
        )
        raise ValueError(
            "Output names do not match between reference model and Core ML exported model: "
            f"{coreml_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        logger.info(f"\t-[✓] Core ML model output names match reference model ({coreml_outputs_set})")

    # Check the shape and values match
    for name in coreml_output_internal_names:
        coreml_name = output_descs[name].name
        coreml_value = coreml_outputs[coreml_name]

        if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
            ref_value = ref_outputs_dict[name].detach().numpy()
        else:
            ref_value = ref_outputs_dict[name].numpy()

        if output_descs[name].do_softmax:
            axis = 1 if config.task == "semantic-segmentation" else -1
            ref_value = softmax(ref_value, axis=axis)

        logger.info(f'\t- Validating Core ML model output "{name}":')

        # Shape
        if not coreml_value.shape == ref_value.shape:
            if config.task == "semantic-segmentation" and (output_descs[name].do_upsample or output_descs[name].do_argmax):
                logger.info("\t\t-[ ] cannot compare outputs because of do_upsample or do_argmax options")
                continue
            else:
                logger.info(f"\t\t-[x] shape {coreml_value.shape} doesn't match {ref_value.shape}")
                raise ValueError(
                    "Output shape doesn't match between reference model and Core ML exported model: "
                    f"Got {ref_value.shape} (reference) and {coreml_value.shape} (Core ML)"
                )
        else:
            logger.info(f"\t\t-[✓] {coreml_value.shape} matches {ref_value.shape}")

        # Values
        if not np.allclose(ref_value, coreml_value, atol=atol):
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Output values do not match between reference model and Core ML exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - coreml_value))}"
            )
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")
