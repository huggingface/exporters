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
"""Core ML conversion for SqueezeBERT."""

import numpy as np
import torch
from torch import nn

import coremltools as ct
from coremltools.models.neural_network import quantization_utils

from transformers import (
    PreTrainedTokenizerBase,
    SqueezeBertModel,
    SqueezeBertForMaskedLM,
    SqueezeBertForMultipleChoice,
    SqueezeBertForQuestionAnswering,
    SqueezeBertForSequenceClassification,
    SqueezeBertForTokenClassification,
)
from ..coreml_utils import *


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False,
        )

        if isinstance(self.model, (
            SqueezeBertForMaskedLM,
            SqueezeBertForMultipleChoice,
            SqueezeBertForSequenceClassification,
            SqueezeBertForTokenClassification,
        )):
            return nn.functional.softmax(outputs[0], dim=-1)  # logits

        if isinstance(self.model, SqueezeBertForQuestionAnswering):
            start_scores = nn.functional.softmax(outputs[0], dim=-1)  # start_logits
            end_scores   = nn.functional.softmax(outputs[1], dim=-1)  # end_logits
            return start_scores, end_scores

        if isinstance(self.model, SqueezeBertModel):
            return outputs[0], outputs[1]  # last_hidden_state, pooler_output

        return None


def export(
    torch_model,
    tokenizer: PreTrainedTokenizerBase,
    sequence_length: int = 64,
    use_attention_mask: bool = True,
    quantize: str = "float32",
    legacy: bool = False,
    **kwargs,
) -> ct.models.MLModel:
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    use_token_type_ids = False
    if isinstance(torch_model, SqueezeBertForMultipleChoice):
        use_token_type_ids = True
        use_attention_mask = True

    if isinstance(torch_model, SqueezeBertForMultipleChoice):
        shape = (1, torch_model.config.num_labels, sequence_length)
    else:
        shape = (1, sequence_length)

    input_ids = torch.randint(tokenizer.vocab_size, shape)
    attention_mask = torch.ones(shape, dtype=torch.int64)
    token_type_ids = torch.zeros(shape, dtype=torch.int64)

    example_input = [ input_ids ]
    if use_attention_mask:
        example_input.append(attention_mask)
    if use_token_type_ids:
        example_input.append(token_type_ids)

    wrapper = Wrapper(torch_model).eval()
    traced_model = torch.jit.trace(wrapper, example_input, strict=True)

    # Run the PyTorch model, to get the shapes of the output tensors.
    with torch.no_grad():
        example_output = traced_model(*example_input)

    convert_kwargs = {}
    if not legacy:
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    if isinstance(torch_model, (SqueezeBertForSequenceClassification, SqueezeBertForMultipleChoice)):
        class_labels = [torch_model.config.id2label[x] for x in range(torch_model.config.num_labels)]
        classifier_config = ct.ClassifierConfig(class_labels)
        convert_kwargs['classifier_config'] = classifier_config

    # pass any additional arguments to ct.convert()
    for key, value in kwargs.items():
        convert_kwargs[key] = value

    input_tensors = [ ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32) ]
    if use_attention_mask:
        input_tensors.append(ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32))
    if use_token_type_ids:
        input_tensors.append(ct.TensorType(name="token_type_ids", shape=token_type_ids.shape, dtype=np.int32))

    mlmodel = ct.convert(
        traced_model,
        inputs=input_tensors,
        convert_to="neuralnetwork" if legacy else "mlprogram",
        **convert_kwargs,
    )

    spec = mlmodel._spec

    user_defined_metadata = {}
    if torch_model.config.transformers_version:
        user_defined_metadata["transformers_version"] = torch_model.config.transformers_version

    mlmodel.input_description["input_ids"] = "Indices of input sequence tokens in the vocabulary"
    if use_attention_mask:
        mlmodel.input_description["attention_mask"] = "Mask to avoid performing attention on padding token indices (1 = not masked, 0 = masked)"
    if use_token_type_ids:
        mlmodel.input_description["token_type_ids"] = "Segment token indices to indicate first and second portions of the inputs (0 = sentence A, 1 = sentence B)"

    if isinstance(torch_model, SqueezeBertForMaskedLM):
        # Rename the output and fill in its shape.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "token_scores")
        set_multiarray_shape(output, example_output.shape)
        mlmodel.output_description["token_scores"] = "Prediction scores for each vocabulary token (after softmax)"

    if isinstance(torch_model, SqueezeBertForQuestionAnswering):
        # Rename the outputs and fill in their shapes.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "start_scores")
        set_multiarray_shape(output, (1, sequence_length))

        output = spec.description.output[1]
        ct.utils.rename_feature(spec, output.name, "end_scores")
        set_multiarray_shape(output, (1, sequence_length))

        mlmodel.output_description["start_scores"] = "Span-start scores (after softmax)"
        mlmodel.output_description["end_scores"] = "Span-end scores (after softmax)"

    if isinstance(torch_model, (
        SqueezeBertForSequenceClassification,
        SqueezeBertForMultipleChoice,
    )):
        probs_output_name = spec.description.predictedProbabilitiesName
        ct.utils.rename_feature(spec, probs_output_name, "probabilities")
        spec.description.predictedProbabilitiesName = "probabilities"

        mlmodel.output_description["probabilities"] = "Probability of each category"
        mlmodel.output_description["classLabel"] = "Category with the highest score"

    if isinstance(torch_model, SqueezeBertForTokenClassification):
        # Rename the output and fill in its shape.
        output = spec.description.output[0]
        ct.utils.rename_feature(spec, output.name, "token_scores")
        set_multiarray_shape(output, example_output.shape)
        mlmodel.output_description["token_scores"] = "Classification scores (after softmax)"

        # Add the class labels to the metadata.
        labels = get_labels_as_list(torch_model)
        user_defined_metadata["classes"] = ",".join(labels)

    if isinstance(torch_model, SqueezeBertModel):
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
