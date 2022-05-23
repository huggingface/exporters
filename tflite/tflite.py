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
"""TensorFlow Lite conversion for Hugging Face Transformers models."""

from transformers import GPT2LMHeadModel
from transformers import ViTModel, ViTForImageClassification
from transformers import MobileViTModel, MobileViTForImageClassification, MobileViTForSemanticSegmentation
from transformers.utils import logging


def export(model, **kwargs):
    """Convert the Hugging Face Transformers model to TensorFlow Lite format.

    Args:
        model:
            A trained TensorFlow model.

    Additional arguments depend on the model.

    Return:
        `TODO`: the converted TensorFlow Lite model
    """
    logger = logging.get_logger(__name__)

    model_type = type(model)

    # if model_type == GPT2LMHeadModel:
    #     from .models import gpt2
    #     return gpt2.export(model, preprocessor, **kwargs)

    logger.warning("Cannot convert unknown model type: " + str(model_type))
    return None
