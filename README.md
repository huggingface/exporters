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

## How to use the Core ML exporter

First, load the model from a checkpoint:

```python
from transformers import ViTForImageClassification, ViTFeatureExtractor
model_checkpoint = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_checkpoint)
torch_model = ViTForImageClassification.from_pretrained(model_checkpoint, torchscript=True)
```

Note: For the best results, pass the argument `torchscript=True` to `from_pretrained` when loading the model. This allows the model to configure itself for PyTorch tracing, which is needed for the Core ML conversion.

Exporting the model to Core ML:

```python
from exporters import coreml
mlmodel = coreml.export(torch_model, feature_extractor=feature_extractor, quantize="float16")
```

Optionally fill in the model's metadata:

```python
mlmodel.short_description = "Your awesome model"
mlmodel.author = "Your name"
mlmodel.license = "Copyright by you"
mlmodel.version = "1.0"
```

Finally, save the model. You can open the resulting **mlpackage** file in Xcode and examine it there.

```python
mlmodel.save("ViT.mlpackage")
```

### Configuring the export options

The arguments to `coreml.export()` are:

- `model` (required): a PyTorch or TensorFlow model instance from the 🤗 Transformers library
- `quantize` (optional): Whether to quantize the model weights. The possible quantization options are: `"float32"` (no quantization) or `"float16"` (for 16-bit floating point).
- `legacy` (optional): By default, a model is generated in the ML Program format. By setting `legacy=True`, the older NeuralNetwork format will be used.
- Model-specific arguments. For image models, this usually includes the `FeatureExtractor` object. Text models will need the sequence length. See below for which arguments to use for your model.
- Any additional arguments are passed straight through to `coremltools.convert`. For example, `compute_units=coremltools.ComputeUnit.CPU_AND_GPU`.

**Note:** It's normal for the conversion process to output warning messages. You can safely ignore these. As long as the output from `coreml.export()` is a `MLModel` object, the conversion was successful. If the conversion failed, `coreml.export()` returns `None` or raises an exception. That said, it's always a good idea to run the original model and the Core ML model on the same inputs, and verify that the outputs are identical or at least have a maximum error of less than 1e-5 or so.

### Verifying the export was successful

When doing the Core ML export on a Mac, it's possible to make predictions from Python using the exported model. For example:

```python
input_ids = np.array([ ... ], dtype=np.int32)
outputs = mlmodel.predict({"input_ids": input_ids})
```

Vision models take a `PIL` image as input:

```python
import requests, PIL.Image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = PIL.Image.open(requests.get(url, stream=True).raw)
image_resized = image.resize((224, 224))

outputs = mlmodel.predict({"image": image_resized})
print(outputs["classLabel"])
```

This is useful for verifying that the exported model indeed works as expected!

## How to use the TensorFlow Lite exporter

To export a model to TF Lite:

```python
TODO
```

## Pushing the model to the Hugging Face Hub

The [Hugging Face Hub](https://huggingface.co) can also host your Core ML and TF Lite models. You can use the [`huggingface_hub` package](https://huggingface.co/docs/huggingface_hub/main/en/index) to upload the converted model to the Hub from Python.

First log in to your Hugging Face account account with the following command:

```bash
huggingface-cli login
```

Alternatively, if you prefer working from a Jupyter or Colaboratory notebook, log in with `notebook_login()`. This will launch a widget in your notebook from which you can enter your Hugging Face credentials.

```python
from huggingface_hub import notebook_login
notebook_login()
```

Once you are logged in, save the **mlpackage** to the Hub as follows:

```python
from huggingface_hub import Repository

with Repository(
        "<model name>", clone_from="https://huggingface.co/<user>/<model name>",
        use_auth_token=True).commit(commit_message="add Core ML model"):
    mlmodel.save("<model name>.mlpackage")
```

Make sure to replace `<model name>` with the name of the model and `<user>` with your Hugging Face username.

## Supported models

Currently, the following PyTorch models can be exported:

| Model | Types | Core ML |
|-------|-------| --------|
| [BERT](https://huggingface.co/docs/transformers/main/model_doc/bert) | `BertForQuestionAnswering` | ✅ |
| [ConvNeXT](https://huggingface.co/docs/transformers/main/model_doc/convnext) | `ConvNextModel`, `ConvNextForImageClassification` | ✅ |
| [CvT](https://huggingface.co/docs/transformers/main/model_doc/cvt) | `CvtModel`, `CvtForImageClassification` | ✅ |
| [DistilBERT](https://huggingface.co/docs/transformers/main/model_doc/distilbert) | `DistilBertForQuestionAnswering`, `DistilBertForSequenceClassification` | ✅ |
| [MobileBERT](https://huggingface.co/docs/transformers/main/model_doc/mobilebert) | `MobileBertModel`, `MobileBertForMaskedLM`, `MobileBertForMultipleChoice`, `MobileBertForNextSentencePrediction`, `MobileBertForPreTraining`, `MobileBertForQuestionAnswering`, `MobileBertForSequenceClassification`, `MobileBertForTokenClassification` | ✅ |
| [MobileViT](https://huggingface.co/docs/transformers/main/model_doc/mobilevit) | `MobileViTModel`, `MobileViTForImageClassification`, `MobileViTForSemanticSegmentation` | ✅ |
| [OpenAI GPT2](https://huggingface.co/docs/transformers/main/model_doc/gpt2), [DistilGPT2](https://huggingface.co/distilgpt2) | `GPT2LMHeadModel` | ✅ |
| [SegFormer](https://huggingface.co/docs/transformers/main/model_doc/segformer) | `SegformerModel`, `SegformerForImageClassification`, `SegformerForSemanticSegmentation` | ✅ |
| [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/main/model_doc/vit) | `ViTModel`, `ViTForImageClassification` | ✅ |

The following TensorFlow models can be exported:

| Model | Types | Core ML | TF Lite |
|-------|-------| --------|---------|
| [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/main/model_doc/vit) | TODO | ❌ | ❌ |

Note: Only TensorFlow models can be exported to TF Lite. PyTorch models are not supported.

## Unsupported models

The following models are known to give errors when attempting conversion to Core ML format:

- [DETR](https://huggingface.co/docs/transformers/main/model_doc/detr). The conversion completes without errors but the Core ML compiler cannot load the model.
- [Data2VecAudio](https://huggingface.co/docs/transformers/main/en/model_doc/data2vec). Same issue as DETR.
- [Speech2Text](https://huggingface.co/docs/transformers/main/en/model_doc/speech_to_text). The "glu" op is not supported by coremltools. Should be possible to solve by defining a `@register_torch_op` function.
- [Swin Transformer](https://huggingface.co/docs/transformers/main/model_doc/swin). The PyTorch graph contains unsupported operations: remainder, roll, adaptive_avg_pool1d.
- [Wav2Vec2](https://huggingface.co/docs/transformers/main/en/model_doc/wav2vec2), [HuBERT](https://huggingface.co/docs/transformers/main/en/model_doc/hubert), [SEW](https://huggingface.co/docs/transformers/main/en/model_doc/sew): Unsupported op for `nn.GroupNorm` (should be possible to solve), invalid broadcasting operations (will be harder to solve), and most likely additional issues.
- [UniSpeech](https://huggingface.co/docs/transformers/main/en/model_doc/unispeech). Missing op for `_weight_norm` (possible to work around), also same Core ML compiler error as DETR.
- [WavLM](https://huggingface.co/docs/transformers/main/en/model_doc/wavlm). Missing ops for `_weight_norm`, `add_`, `full_like`.

## Model-specific conversion options

Pass these additional options into `coreml.export()` or `tflite.export()`.

### BERT, DistilBERT

- `tokenizer` (required). The `(Distil)BertTokenizer` object for the trained model.
- `sequence_length` (required). The input tensor has shape `(batch, sequence length)`. In the exported model, the sequence length will be a fixed number. The default sequence length is 128.

### ConvNeXT, CvT

- `feature_extractor` (required). The `ConvNextFeatureExtractor` object for the trained model.

### MobileBERT

- `tokenizer` (required). The `PreTrainedTokenizer` object for the trained model.
- `sequence_length` (required). The input tensor has shape `(batch, sequence length)`. In the exported model, the sequence length will be a fixed number. The default sequence length is 128.
- `use_attention_mask` (optional, default is `True`). If true, adds an `attention_mask` input to the exported model.

### MobileViT

- `feature_extractor` (required). The `MobileViTFeatureExtractor` object for the trained model.
- `do_upsample` (optional, default is `True`). Scales the output to have the same width and height as the input.
- `do_argmax` (optional, default is `True`). Whether to perform an argmax operation on the predicted logits.

### OpenAI GPT2, DistilGPT2

- `tokenizer` (required). The `GPT2Tokenizer` object for the trained model.
- `sequence_length` (required). The input tensor has shape `(batch, sequence length, vocab size)`. In the exported model, the sequence length will be a fixed number. The default sequence length is 64.
- `legacy=True`. This model needs to be exported as NeuralNetwork; it currently does not work correctly as ML Program.

### SegFormer

- `feature_extractor` (required). The `SegformerFeatureExtractor` object for the trained model.
- `do_upsample` (optional, default is `True`). Scales the output to have the same width and height as the input.
- `do_argmax` (optional, default is `True`). Whether to perform an argmax operation on the predicted logits.

### ViT

- `feature_extractor` (required). The `ViTFeatureExtractor` object for the trained model.

## Additional notes

### Core ML

The `exporters.coreml` module uses the [coremltools](https://coremltools.readme.io/docs) package to perform the conversion from PyTorch or TensorFlow to Core ML format.

The exported Core ML models use the **mlpackage** format with the **ML Program** model type. This format was introduced in 2021 and requires at least iOS 15, macOS 12.0, and Xcode 13. We prefer to use this format as it is the future of Core ML. Unfortunately, for some models the generated ML Program is incorrect, in which case it's recommended to convert the model to the older NeuralNetwork format by passing in the argument `legacy=True`. On certain hardware, the older format may also run more efficiently. If you're not sure which one to use, export the model twice and compare the two versions.

Additional notes:

- Image models will automatically perform image preprocessing as part of the model. You do not need to preprocess the image yourself, except potentially resizing or cropping it.

- Text models will require manual tokenization of the input data. Core ML does not have its own tokenization support.

- For classification models, a softmax layer is added during the conversion process and the labels are included in the `MLModel` object.

- For models that output logits, a softmax layer is usually added during the conversion process to convert the logits into probabilities. However, for certain models (e.g. GPT2) this was found to result in NaN values in the output. For these models where softmax was problematic, it was left out.

- For semantic segmentation and object detection models, the labels are included in the `MLModel` object's metadata.

- For semantic segmentation models, the model takes the argmax and the outputs are the predicted class label indices. This can be disabled, in which case the outputs are the predicted class probabilities. Note: With the argmax enabled, the Preview pane won't be available in Xcode's model viewer.

- ML Programs currently only support 16-bit float quantization, not integer quantization. This is a limitation of Core ML.

- Image classifier models have the usual `classLabel` and `probabilities` outputs, but also a "hidden" `var_xxx` output with the softmax results. This appears to be a minor bug in the converter; it doesn't hurt anything to keep this extra output.

### TensorFlow Lite

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
