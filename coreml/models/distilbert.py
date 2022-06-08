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
"""Core ML conversion for DistilBERT and BERT."""

import numpy as np
import torch
from torch import nn

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from transformers import (
    PreTrainedTokenizerBase,
    BertForQuestionAnswering,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification
)
from ..coreml_utils import *


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, inputs):
        outputs = self.model(inputs, return_dict=False)

        if is_any_instance(self.model, [BertForQuestionAnswering, DistilBertForQuestionAnswering]):
            start_logits = outputs[0]
            end_logits = outputs[1]
            start_scores = nn.functional.softmax(start_logits, dim=1)
            end_scores = nn.functional.softmax(end_logits, dim=1)
            return start_scores, end_scores

        if isinstance(self.model, DistilBertForSequenceClassification):
            return nn.functional.softmax(outputs[0], dim=1)

        return None


def export(
    torch_model,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int = 64,
    quantize: str = "float32",
    legacy: bool = False,
) -> ct.models.MLModel:
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    example_input = torch.randint(tokenizer.vocab_size, (1, sequence_length))

    wrapper = Wrapper(torch_model).eval()
    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    convert_kwargs = {}
    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    if isinstance(torch_model, DistilBertForSequenceClassification):
        class_labels = [torch_model.config.id2label[x] for x in range(torch_model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=example_input.shape, dtype=np.int32)],
        convert_to="neuralnetwork" if legacy else "mlprogram",
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if torch_model.config.transformers_version:
        user_defined_metadata["transformers_version"] = torch_model.config.transformers_version

    mlmodel.input_description["input_ids"] = "Indices of input sequence tokens in the vocabulary"

    if isinstance(torch_model, DistilBertForSequenceClassification):
        probs_output_name = spec.description.predictedProbabilitiesName
        ct.utils.rename_feature(spec, probs_output_name, "probabilities")
        spec.description.predictedProbabilitiesName = "probabilities"

        mlmodel.output_description["probabilities"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if is_any_instance(torch_model, [BertForQuestionAnswering, DistilBertForQuestionAnswering]):
        # Rename the outputs and fill in their shapes.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "start_scores")
        set_multiarray_shape(output, (1, sequence_length))

        output = spec.description.output[1]
        ct.utils.rename_feature(spec, output.name, "end_scores")
        set_multiarray_shape(output, (1, sequence_length))

        mlmodel.output_description["start_scores"] = "Span-start scores (after softmax)"
        mlmodel.output_description["end_scores"] = "Span-end scores (after softmax)"

    if len(user_defined_metadata) > 0:
        spec.description.metadata.userDefined.update(user_defined_metadata)

    # Reload the model in case any input / output names were changed.
    mlmodel = ct.models.MLModel(mlmodel._spec, weights_dir=mlmodel.weights_dir)

    if legacy and quantize == "float16":
        mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=16)

    return mlmodel
