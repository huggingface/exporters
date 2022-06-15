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
"""Core ML conversion for MobileBERT."""

import numpy as np
import torch
from torch import nn

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from transformers import (
    PreTrainedTokenizerBase,
    MobileBertModel,
    MobileBertForMaskedLM,
    MobileBertForMultipleChoice,
    MobileBertForNextSentencePrediction,
    MobileBertForPreTraining,
    MobileBertForQuestionAnswering,
    MobileBertForSequenceClassification,
    MobileBertForTokenClassification,
)
from ..coreml_utils import *


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, inputs):
        outputs = self.model(inputs, return_dict=False)

        if isinstance(self.model, MobileBertForPreTraining):
            return outputs[0], outputs[1]  # prediction_logits, seq_relationship_logits

        if is_any_instance(self.model, [
            MobileBertForMaskedLM,
            MobileBertForMultipleChoice,
            MobileBertForNextSentencePrediction,
            MobileBertForSequenceClassification,
            MobileBertForTokenClassification,
        ]):
            return nn.functional.softmax(outputs[0], dim=-1)

        if isinstance(self.model, MobileBertForQuestionAnswering):
            start_logits = outputs[0]
            end_logits = outputs[1]
            start_scores = nn.functional.softmax(start_logits, dim=-1)
            end_scores = nn.functional.softmax(end_logits, dim=-1)
            return start_scores, end_scores

        if isinstance(self.model, MobileBertModel):
            return outputs[0], outputs[1]  # last_hidden_state, pooler_output

        return None


def export(
    torch_model,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int = 64,
    quantize: str = "float32",
    legacy: bool = False,
    **kwargs,
) -> ct.models.MLModel:
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    if isinstance(torch_model, MobileBertForMultipleChoice):
        example_input = torch.randint(tokenizer.vocab_size, (1, torch_model.config.num_labels, sequence_length))
    else:
        example_input = torch.randint(tokenizer.vocab_size, (1, sequence_length))

    wrapper = Wrapper(torch_model).eval()
    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    convert_kwargs = {}
    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    if is_any_instance(torch_model, [MobileBertForSequenceClassification, MobileBertForMultipleChoice]):
        class_labels = [torch_model.config.id2label[x] for x in range(torch_model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    if isinstance(torch_model, MobileBertForNextSentencePrediction):
        class_labels = ["true", "false"]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    # pass any additional arguments to ct.convert()
    for key, value in kwargs.items():
        convert_kwargs[key] = value

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=example_input.shape, dtype=np.int32)],
        convert_to="neuralnetwork" if legacy else "mlprogram",
        **convert_kwargs,
    )

    # Run the PyTorch model, to get the shapes of the output tensors.
    with torch.no_grad():
        example_output = traced_model(example_input)

    spec = mlmodel._spec

    user_defined_metadata = {}
    if torch_model.config.transformers_version:
        user_defined_metadata["transformers_version"] = torch_model.config.transformers_version

    mlmodel.input_description["input_ids"] = "Indices of input sequence tokens in the vocabulary"

    if isinstance(torch_model, MobileBertForMaskedLM):
        # Rename the output and fill in its shape.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "scores")
        set_multiarray_shape(output, example_output.shape)
        mlmodel.output_description["scores"] = "Prediction scores for each vocabulary token (after softmax)"

    if isinstance(torch_model, MobileBertForPreTraining):
        # Rename the outpust and fill in their shapes.
        ct.utils.rename_feature(spec, spec.description.output[0].name, "prediction_logits")
        ct.utils.rename_feature(spec, spec.description.output[1].name, "seq_relationship_logits")

        set_multiarray_shape(get_output_named(spec, "prediction_logits"), example_output[0].shape)
        set_multiarray_shape(get_output_named(spec, "seq_relationship_logits"), example_output[1].shape)

        mlmodel.output_description["prediction_logits"] = "Prediction scores of the language modeling head (before softmax)"
        mlmodel.output_description["seq_relationship_logits"] = "Prediction scores of the next sequence prediction head (before softmax)"

    if isinstance(torch_model, MobileBertForQuestionAnswering):
        # Rename the outputs and fill in their shapes.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "start_scores")
        set_multiarray_shape(output, (1, sequence_length))

        output = spec.description.output[1]
        ct.utils.rename_feature(spec, output.name, "end_scores")
        set_multiarray_shape(output, (1, sequence_length))

        mlmodel.output_description["start_scores"] = "Span-start scores (after softmax)"
        mlmodel.output_description["end_scores"] = "Span-end scores (after softmax)"

    if is_any_instance(torch_model, [
        MobileBertForSequenceClassification,
        MobileBertForNextSentencePrediction,
        MobileBertForMultipleChoice,
    ]):
        probs_output_name = spec.description.predictedProbabilitiesName
        ct.utils.rename_feature(spec, probs_output_name, "probabilities")
        spec.description.predictedProbabilitiesName = "probabilities"

        mlmodel.output_description["probabilities"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if isinstance(torch_model, MobileBertForTokenClassification):
        # Rename the output and fill in its shape.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "scores")
        set_multiarray_shape(output, example_output.shape)
        mlmodel.output_description["scores"] = "Classification scores (after softmax)"

        # Add the class labels to the metadata.
        labels = get_labels_as_list(torch_model)
        user_defined_metadata["classes"] = ",".join(labels)

    if isinstance(torch_model, MobileBertModel):
        # Rename the outputs.
        ct.utils.rename_feature(spec, "hidden_states", "last_hidden_state")

        output_names = get_output_names(spec)
        for output_name in output_names:
            if output_name != "last_hidden_state":
                ct.utils.rename_feature(spec, output_name, "pooler_output")

        set_multiarray_shape(get_output_named(spec, "last_hidden_state"), example_output[0].shape)
        set_multiarray_shape(get_output_named(spec, "pooler_output"), example_output[1].shape)

        mlmodel.output_description["last_hidden_state"] = "Hidden states from the last layer"
        mlmodel.output_description["pooler_output"] = "Last layer hidden-state of the first token of the sequence"

    if len(user_defined_metadata) > 0:
        spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    if legacy and quantize == "float16":
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    return mlmodel
