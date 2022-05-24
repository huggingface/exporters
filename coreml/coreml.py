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
"""Core ML conversion for Hugging Face Transformers models."""

from typing import Optional

import coremltools as ct

from transformers import BertForQuestionAnswering
from transformers import DistilBertForQuestionAnswering
from transformers import GPT2LMHeadModel
from transformers import MobileViTModel, MobileViTForImageClassification, MobileViTForSemanticSegmentation
from transformers import ViTModel, ViTForImageClassification
from transformers.utils import logging


def export(model, quantize: str = "float32", **kwargs) -> Optional[ct.models.MLModel]:
    """Convert the Hugging Face Transformers model to Core ML format.

    Args:
        model:
            A trained PyTorch or TensorFlow model.
        quantize (`str`, *optional*, defaults to `"float32"`):
            Quantization options. Possible values: `"float32"`, `"float16"`.

    Additional arguments depend on the model.

    Return:
        `coremltools.models.MLModel`: the converted Core ML model
    """
    logger = logging.get_logger(__name__)

    model_type = type(model)

    kwargs["quantize"] = quantize

    if model_type in [BertForQuestionAnswering, DistilBertForQuestionAnswering]:
        from .models import distilbert
        return distilbert.export(model, **kwargs)

    elif model_type == GPT2LMHeadModel:
        from .models import gpt2
        return gpt2.export(model, **kwargs)

    elif model_type in [MobileViTModel, MobileViTForImageClassification, MobileViTForSemanticSegmentation]:
        from .models import mobilevit
        return mobilevit.export(model, **kwargs)

    elif model_type in [ViTModel, ViTForImageClassification]:
        from .models import vit
        return vit.export(model, **kwargs)

    logger.warning("Cannot convert unknown model type: " + str(model_type))
    return None
