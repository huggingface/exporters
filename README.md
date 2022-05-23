<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🤗 Exporters

👷 **WORK IN PROGRESS** 👷

This package lets you export 🤗 Transformers models to Core ML and TensorFlow Lite.

## When to use Exporters

🤗 Transformers models are implemented in PyTorch, TensorFlow, or JAX. However, for deployment you might want to use a different framework such as Core ML or TensorFlow Lite. This library makes it easy to convert the models to these formats.

The aim of the Exporters package is to be more convenient than writing your own conversion script with *coremltools* or *TFLiteConverter*, and to be tightly integrated with the 🤗 Transformers library and the Hugging Face Hub.

Note: Keep in mind that Transformer models are usually quite large and are not always suitable for use on mobile devices. It might be a good idea to [optimize the model for inference](https://github.com/huggingface/optimum) first using 🤗 Optimum.

## How to use exporters

### Core ML

To export a model to Core ML:

```python
from transformers import ViTForImageClassification, ViTFeatureExtractor
model_checkpoint = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_checkpoint)
torch_model = ViTForImageClassification.from_pretrained(model_checkpoint)

from exporters import coreml
mlmodel = coreml.export(torch_model, preprocessor=feature_extractor, quantize="float32")
mlmodel.save("ViT.mlpackage")
```

Optionally fill in the model's metadata:

```python
mlmodel.short_description = "Your awesome model"
mlmodel.author = "Your name"
mlmodel.license = "Copyright by you"
mlmodel.version = "1.0"
```

You can add the resulting **mlpackage** file to your Xcode project and examine it there. 

It's also possible to make predictions from Python using the exported model. For example:

```python
import requests, PIL.Image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = PIL.Image.open(requests.get(url, stream=True).raw)
image_resized = image.resize((256, 256))

outputs = mlmodel.predict({"image": image_resized})
print(outputs["classLabel"])
```

The arguments to `coreml.export()` are:

- `model` (required): a PyTorch or TensorFlow model instance from the 🤗 Transformers library
- `quantize` (optional): Whether to quantize the model weights. The possible quantization options are: `"float32"` (no quantization) or `"float16"` (for 16-bit floating point).
- any model-specific arguments

For image models, the model-specific arguments usually include the `FeatureExtractor` object. Text models will need the sequence length. See below for which arguments to use for your model.

## TensorFlow Lite

To export a model to TF Lite:

```python
TODO
```

## Pushing the model to the Hugging Face Hub

The [Hugging Face Hub](https://huggingface.co) can also host your Core ML and TF Lite models.

To push the converted model to the Hugging Face Hub:

```python
TODO
```

## Supported models

Currently, the following PyTorch models can be exported:

| Model | Types | Core ML |
|-------|-------| --------|
| [MobileViT](https://huggingface.co/docs/transformers/main/model_doc/mobilevit) | `MobileViTModel`, `MobileViTForImageClassification`, `MobileViTForSemanticSegmentation` | ✅ |
| [OpenAI GPT2](https://huggingface.co/docs/transformers/main/model_doc/gpt2) | `GPT2LMHeadModel` | ✅ |
| [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/main/model_doc/vit) | `ViTModel`, `ViTForImageClassification` | ✅ |

The following TensorFlow models can be exported:

| Model | Types | Core ML | TF Lite |
|-------|-------| --------|---------|
| [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/main/model_doc/vit) | TODO | ❌ | ❌ |

Note: Only TensorFlow models can be exported to TF Lite. PyTorch models are not supported.

## Model-specific conversion options

Pass these additional options into `coreml.export()` or `tflite.export()`.

### MobileViT

- `preprocessor`. The `MobileViTFeatureExtractor` object for the trained model.

### OpenAI GPT2

- `sequence_length`. The input tensor has shape `(batch, sequence length, vocab size)`. In the exported model, the sequence length will be a fixed number. The default sequence length is 64.

### ViT

- `preprocessor`. The `ViTFeatureExtractor` object for the trained model.

## Exporting to Core ML

The `exporters.coreml` module uses the [coremltools](https://coremltools.readme.io/docs) package to perform the conversion from PyTorch or TensorFlow to Core ML format.

The exported Core ML models use the **mlpackage** format with the **ML Program** model type. This new format was introduced in 2021 and requires at least iOS 15, macOS 12.0, and Xcode 13. While it might still be possible to convert certain models to the older NeuralNetwork format, we do not explicitly support this.

Additional notes:

- The converter returns a `coremltools.models.MLModel` object. You are supposed to save this to a **.mlpackage** file yourself. You can also modify the generated model afterwards with coremltools, for example to rename the inputs and outputs.

- Image models will automatically perform image preprocessing as part of the model.

- Text models will require manual tokenization of the input data. Core ML does not have its own tokenization support.

- For classification models, a softmax layer is automatically added and the labels are included in the `MLModel` object. 

- For semantic segmentation and object detection models, the labels are included in the `MLModel` object's metadata.

- ML Programs currently only support 16-bit float quantization, not integer quantization.

## Exporting to TensorFlow Lite

The `exporters.tflite` module uses the [TFLiteConverter](https://www.tensorflow.org/lite/convert/) package to perform the conversion from TensorFlow to TF Lite format.

## What if your model is not supported?

If the model you wish to export is not currently supported by 🤗 Exporters, you can use [coremltools](https://coremltools.readme.io/docs) or [TFLiteConverter](https://www.tensorflow.org/lite/convert/) to do the conversion yourself. 

**Tip:** Look at the existing conversion code for models similar to yours to see how best to do this conversion. Sometimes it's just a matter of copy-pasting the conversion code.

When running these automated conversion tools, it's quite possible the conversion fails with an inscrutable error message. Usually this happens because the model performs an operation that is not supported by Core ML or TF Lite, but the conversion tools also occasionally have bugs or may choke on complex models.

If coremltools fails, you have a couple of options:

1. Fix the original model. This requires a deep understanding of how the model works and is not trivial. However, sometimes the fix is to hardcode certain values rather than letting PyTorch or TensorFlow calculate them from the shapes of tensors.

2. Fix coremltools itself. It is sometimes possible to hack coremltools so that it ignores the issue.

3. Forget about automated conversion and [build the model from scratch using MIL](https://coremltools.readme.io/docs/model-intermediate-language). This is the intermediate language that coremltools uses internally to represent models. It's similar in many ways to PyTorch.

4. Submit an issue and we'll see what we can do. 😀
