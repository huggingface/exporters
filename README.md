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

This package lets you export 🤗 Transformers models to Core ML.

> For converting models to TFLite, we recommend using [Optimum](https://huggingface.co/docs/optimum/exporters/tflite/usage_guides/export_a_model).

## When to use 🤗 Exporters

🤗 Transformers models are implemented in PyTorch, TensorFlow, or JAX. However, for deployment you might want to use a different framework such as Core ML. This library makes it easy to convert Transformers models to this format.

The aim of the Exporters package is to be more convenient than writing your own conversion script with *coremltools* and to be tightly integrated with the 🤗 Transformers library and the Hugging Face Hub.

For an even more convenient approach, `Exporters` powers a [no-code transformers to Core ML conversion Space](https://huggingface.co/spaces/huggingface-projects/transformers-to-coreml). You can try it out without installing anything to check whether the model you are interested in can be converted. If conversion succeeds, the converted Core ML weights will be pushed to the Hub. For additional flexibility and details about the conversion process, please read on.

Note: Keep in mind that Transformer models are usually quite large and are not always suitable for use on mobile devices. It might be a good idea to [optimize the model for inference](https://github.com/huggingface/optimum) first using 🤗 Optimum.

## Installation

Clone this repo:

```bash
$ git clone https://github.com/huggingface/exporters.git
```

Install it as a Python package:

```bash
$ cd exporters
$ pip install -e .
```

All done!

Note: The Core ML exporter can be used from Linux but macOS is recommended.

## Core ML

[Core ML](https://developer.apple.com/machine-learning/core-ml/) is Apple's software library for fast on-device model inference with neural networks and other types of machine learning models. It can be used on macOS, iOS, tvOS, and watchOS, and is optimized for using the CPU, GPU, and Apple Neural Engine. Although the Core ML framework is proprietary, the Core ML file format is an open format.

The Core ML exporter uses [coremltools](https://coremltools.readme.io/docs) to perform the conversion from PyTorch or TensorFlow to Core ML.

The `exporters.coreml` package enables you to convert model checkpoints to a Core ML model by leveraging configuration objects. These configuration objects come ready-made for a number of model architectures, and are designed to be easily extendable to other architectures.

Ready-made configurations include the following architectures:

- BEiT
- BERT
- ConvNeXT
- CTRL
- CvT
- DistilBERT
- DistilGPT2
- GPT2
- LeViT
- MobileBERT
- MobileViT
- SegFormer
- SqueezeBERT
- Vision Transformer (ViT)
- YOLOS

<!-- TODO: automatically generate this list -->

[See here](MODELS.md) for a complete list of supported models.

### Exporting a model to Core ML

<!--
To export a 🤗 Transformers model to Core ML, you'll first need to install some extra dependencies:

``bash
pip install transformers[coreml]
``

The `transformers.coreml` package can then be used as a Python module:
-->

The `exporters.coreml` package can be used as a Python module from the command line. To export a checkpoint using a ready-made configuration, do the following:

```bash
python -m exporters.coreml --model=distilbert-base-uncased exported/
```

This exports a Core ML version of the checkpoint defined by the `--model` argument. In this example it is `distilbert-base-uncased`, but it can be any checkpoint on the Hugging Face Hub or one that's stored locally.

The resulting Core ML file will be saved to the `exported` directory as `Model.mlpackage`. Instead of a directory you can specify a filename, such as `DistilBERT.mlpackage`.

It's normal for the conversion process to output many warning messages and other logging information. You can safely ignore these. If all went well, the export should conclude with the following logs:

```bash
Validating Core ML model...
	-[✓] Core ML model output names match reference model ({'last_hidden_state'})
	- Validating Core ML model output "last_hidden_state":
		-[✓] (1, 128, 768) matches (1, 128, 768)
		-[✓] all values close (atol: 0.0001)
All good, model saved at: exported/Model.mlpackage
```

Note: While it is possible to export models to Core ML on Linux, the validation step will only be performed on Mac, as it requires the Core ML framework to run the model.

The resulting file is `Model.mlpackage`. This file can be added to an Xcode project and be loaded into a macOS or iOS app.

The exported Core ML models use the **mlpackage** format with the **ML Program** model type. This format was introduced in 2021 and requires at least iOS 15, macOS 12.0, and Xcode 13. We prefer to use this format as it is the future of Core ML. The Core ML exporter can also make models in the older `.mlmodel` format, but this is not recommended.

The process is identical for TensorFlow checkpoints on the Hub. For example, you can export a pure TensorFlow checkpoint from the [Keras organization](https://huggingface.co/keras-io) as follows:

```bash
python -m exporters.coreml --model=keras-io/transformers-qa exported/
```

To export a model that's stored locally, you'll need to have the model's weights and tokenizer files stored in a directory. For example, we can load and save a checkpoint as follows:

```python
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> # Load tokenizer and PyTorch weights form the Hub
>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
>>> pt_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
>>> # Save to disk
>>> tokenizer.save_pretrained("local-pt-checkpoint")
>>> pt_model.save_pretrained("local-pt-checkpoint")
```

Once the checkpoint is saved, you can export it to Core ML by pointing the `--model` argument to the directory holding the checkpoint files:

```bash
python -m exporters.coreml --model=local-pt-checkpoint exported/
```

<!--
TODO: also TFAutoModel example
-->

### Selecting features for different model topologies

Each ready-made configuration comes with a set of _features_ that enable you to export models for different types of topologies or tasks. As shown in the table below, each feature is associated with a different auto class:

| Feature                                      | Auto Class                           |
| -------------------------------------------- | ------------------------------------ |
| `default`, `default-with-past`               | `AutoModel`                          |
| `causal-lm`, `causal-lm-with-past`           | `AutoModelForCausalLM`               |
| `ctc`                                        | `AutoModelForCTC`                    |
| `image-classification`                       | `AutoModelForImageClassification`    |
| `masked-im`                                  | `AutoModelForMaskedImageModeling`    |
| `masked-lm`                                  | `AutoModelForMaskedLM`               |
| `multiple-choice`                            | `AutoModelForMultipleChoice`         |
| `next-sentence-prediction`                   | `AutoModelForNextSentencePrediction` |
| `object-detection`                           | `AutoModelForObjectDetection`        |
| `question-answering`                         | `AutoModelForQuestionAnswering`      |
| `semantic-segmentation`                      | `AutoModelForSemanticSegmentation`   |
| `seq2seq-lm`, `seq2seq-lm-with-past`         | `AutoModelForSeq2SeqLM`              |
| `sequence-classification`                    | `AutoModelForSequenceClassification` |
| `speech-seq2seq`, `speech-seq2seq-with-past` | `AutoModelForSpeechSeq2Seq`          |
| `token-classification`                       | `AutoModelForTokenClassification`    |

For each configuration, you can find the list of supported features via the `FeaturesManager`. For example, for DistilBERT we have:

```python
>>> from exporters.coreml.features import FeaturesManager

>>> distilbert_features = list(FeaturesManager.get_supported_features_for_model_type("distilbert").keys())
>>> print(distilbert_features)
['default', 'masked-lm', 'multiple-choice', 'question-answering', 'sequence-classification', 'token-classification']
```

You can then pass one of these features to the `--feature` argument in the `exporters.coreml` package. For example, to export a text-classification model we can pick a fine-tuned model from the Hub and run:

```bash
python -m exporters.coreml --model=distilbert-base-uncased-finetuned-sst-2-english \
                           --feature=sequence-classification exported/
```

which will display the following logs:

```bash
Validating Core ML model...
	- Core ML model is classifier, validating output
		-[✓] predicted class NEGATIVE matches NEGATIVE
		-[✓] number of classes 2 matches 2
		-[✓] all values close (atol: 0.0001)
All good, model saved at: exported/Model.mlpackage
```

Notice that in this case, the exported model is a Core ML classifier, which predicts the highest scoring class name in addition to a dictionary of probabilities, instead of the `last_hidden_state` we saw with the `distilbert-base-uncased` checkpoint earlier. This is expected since the fine-tuned model has a sequence classification head.

<Tip>

The features that have a `with-past` suffix (e.g. `causal-lm-with-past`) correspond to model topologies with precomputed hidden states (key and values in the attention blocks) that can be used for fast autoregressive decoding.

</Tip>

### Configuring the export options

To see the full list of possible options, run the following from the command line:

```bash
python -m exporters.coreml --help
```

Exporting a model requires at least these arguments:

- `-m <model>`: The model ID from the Hugging Face Hub, or a local path to load the model from.
- `--feature <task>`: The task the model should perform, for example `"image-classification"`. See the table above for possible task names.
- `<output>`: The path where to store the generated Core ML model.

The output path can be a folder, in which case the file will be named `Model.mlpackage`, or you can also specify the filename directly.

Additional arguments that can be provided:

- `--preprocessor <value>`: Which type of preprocessor to use. `auto` tries to automatically detect it. Possible values are: `auto` (the default), `tokenizer`, `feature_extractor`, `processor`.
- `--atol <number>`: The absolute difference tolerence used when validating the model. The default value is 1e-4.
- `--quantize <value>`: Whether to quantize the model weights. The possible quantization options are: `float32` for no quantization (the default) or `float16` for 16-bit floating point.
- `--compute_units <value>`: Whether to optimize the model for CPU, GPU, and/or Neural Engine. Possible values are: `all` (the default), `cpu_and_gpu`, `cpu_only`, `cpu_and_ne`.

### Using the exported model

Using the exported model in an app is just like using any other Core ML model. After adding the model to Xcode, it will auto-generate a Swift class that lets you make predictions from within the app.

Depending on the chosen export options, you may still need to preprocess or postprocess the input and output tensors.

For image inputs, there is no need to perform any preprocessing as the Core ML model will already normalize the pixels. For classifier models, the Core ML model will output the predictions as a dictionary of probabilities. For other models, you might need to do more work.

Core ML does not have the concept of a tokenizer and so text models will still require manual tokenization of the input data. [Here is an example](https://github.com/huggingface/swift-coreml-transformers) of how to perform tokenization in Swift.

### Overriding default choices in the configuration object

An important goal of Core ML is to make it easy to use the models inside apps. Where possible, the Core ML exporter will add extra operations to the model, so that you do not have to do your own pre- and postprocessing.

In particular,

- Image models will automatically perform pixel normalization as part of the model. You do not need to preprocess the image yourself, except potentially resizing or cropping it.

- For classification models, a softmax layer is added and the labels are included in the model file. Core ML makes a distinction between classifier models and other types of neural networks. For a model that outputs a single classification prediction per input example, Core ML makes it so that the model predicts the winning class label and a dictionary of probabilities instead of a raw logits tensor. Where possible, the exporter uses this special classifier model type.

- Other models predict logits but do not fit into Core ML's definition of a classifier, such as the `token-classificaton` task that outputs a prediction for each token in the sequence. Here, the exporter also adds a softmax to convert the logits into probabilities. The label names are added to the model's metadata. Core ML ignores these label names but they can be retrieved by writing a few lines of Swift code.

- A `semantic-segmentation` model will upsample the output image to the original spatial dimensions and apply an argmax to obtain the predicted class label indices. It does not automatically apply a softmax.

The Core ML exporter makes these choices because they are the settings you're most likely to need. To override any of the above defaults, you must create a subclass of the configuration object, and then export the model to Core ML by writing a short Python program.

Example: To prevent the MobileViT semantic segmentation model from upsampling the output image, you would create a subclass of `MobileViTCoreMLConfig` and override the `outputs` property to set `do_upsample` to False. Other options you can set for this output are `do_argmax` and `do_softmax`.

```python
from collections import OrderedDict
from exporters.coreml.models import MobileViTCoreMLConfig
from exporters.coreml.config import OutputDescription

class MyCoreMLConfig(MobileViTCoreMLConfig):
    @property
    def outputs(self) -> OrderedDict[str, OutputDescription]:
        return OrderedDict(
            [
                (
                    "logits",
                    OutputDescription(
                        "classLabels",
                        "Classification scores for each pixel",
                        do_softmax=True,
                        do_upsample=False,
                        do_argmax=False,
                    )
                ),
            ]
        )

config = MyCoreMLConfig(model.config, "semantic-segmentation")
```

Here you can also change the name of the output from `classLabels` to something else, or fill in the output description ("Classification scores for each pixel").

It is also possible to change the properties of the model inputs. For example, for text models the default sequence length is between 1 and 128 tokens. To set the input sequence length on a DistilBERT model to a fixed length of 32 tokens, you could override the config object as follows:

```python
from collections import OrderedDict
from exporters.coreml.models import DistilBertCoreMLConfig
from exporters.coreml.config import InputDescription

class MyCoreMLConfig(DistilBertCoreMLConfig):
    @property
    def inputs(self) -> OrderedDict[str, InputDescription]:
        input_descs = super().inputs
        input_descs["input_ids"].sequence_length = 32
        return input_descs

config = MyCoreMLConfig(model.config, "text-classification")
```

Using a fixed sequence length generally outputs a simpler, and possibly faster, Core ML model. However, for many models the input needs to have a flexible length. In that case, specify a tuple for `sequence_length` to set the (min, max) lengths. Use (1, -1) to have no upper limit on the sequence length. (Note: if `sequence_length` is set to a fixed value, then the batch size is fixed to 1.)

To find out what input and output options are available for the model you're interested in, create its `CoreMLConfig` object and examine the `config.inputs` and `config.outputs` properties.

Not all inputs or outputs are always required: For text models, you may remove the `attention_mask` input. Without this input, the attention mask is always assumed to be filled with ones (no padding). However, if the task requires a `token_type_ids` input, there must also be an `attention_mask` input.

Removing inputs and/or outputs is accomplished by making a subclass of `CoreMLConfig` and overriding the `inputs` and `outputs` properties.

By default, a model is generated in the ML Program format. By overriding the `use_legacy_format` property to return `True`, the older NeuralNetwork format will be used. This is not recommended and only exists as a workaround for models that fail to convert to the ML Program format.

Once you have the modified `config` instance, you can use it to export the model following the instructions from the section "Exporting the model" below.

Not everything is described by the configuration objects. The behavior of the converted model is also determined by the model's tokenizer or feature extractor. For example, to use a different input image size, you'd create the feature extractor with different resizing or cropping settings and use that during the conversion instead of the default feature extractor.

### Exporting a model for an unsupported architecture

If you wish to export a model whose architecture is not natively supported by the library, there are three main steps to follow:

1. Implement a custom Core ML configuration.
2. Export the model to Core ML.
3. Validate the outputs of the PyTorch and exported models.

In this section, we'll look at how DistilBERT was implemented to show what's involved with each step.

#### Implementing a custom Core ML configuration

TODO: didn't write this section yet because the implementation is not done yet

Let’s start with the configuration object. We provide an abstract classes that you should inherit from, `CoreMLConfig`.

```python
from exporters.coreml import CoreMLConfig
```

TODO: stuff to cover here:

- `modality` property
- how to implement custom ops + link to coremltools documentation on this topic
- decoder models (`use_past`) and encoder-decoder models (`seq2seq`)

#### Exporting the model

Once you have implemented the Core ML configuration, the next step is to export the model. Here we can use the `export()` function provided by the `exporters.coreml` package. This function expects the Core ML configuration, along with the base model and tokenizer (for text models) or feature extractor (for vision models):

```python
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from exporters.coreml import export
from exporters.coreml.models import DistilBertCoreMLConfig

model_ckpt = "distilbert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, torchscript=True)
preprocessor = AutoTokenizer.from_pretrained(model_ckpt)

coreml_config = DistilBertCoreMLConfig(base_model.config, task="text-classification")
mlmodel = export(preprocessor, base_model, coreml_config)
```

Note: For the best results, pass the argument `torchscript=True` to `from_pretrained` when loading the model. This allows the model to configure itself for PyTorch tracing, which is needed for the Core ML conversion.

Additional options that can be passed into `export()`:

- `quantize`: Use `"float32"` for no quantization (the default), `"float16"` to quantize the weights to 16-bit floats.
- `compute_units`: Whether to optimize the model for CPU, GPU, and/or Neural Engine. Defaults to `coremltools.ComputeUnit.ALL`.

To export the model with precomputed hidden states (key and values in the attention blocks) for fast autoregressive decoding, pass the argument `use_past=True` when creating the `CoreMLConfig` object.

It is normal for the Core ML exporter to print out a lot of warning and information messages. In particular, you might see messages such as these:

> TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!

Those messages are to be expected and are a normal part of the conversion process. If there is a real problem, the converter will throw an error.

If the export succeeded, the return value from `export()` is a `coremltools.models.MLModel` object. Write `print(mlmodel)` to examine the Core ML model's inputs, outputs, and metadata.

Optionally fill in the model's metadata:

```python
mlmodel.short_description = "Your awesome model"
mlmodel.author = "Your name"
mlmodel.license = "Fill in the copyright information here"
mlmodel.version = "1.0"
```

Finally, save the model. You can open the resulting **mlpackage** file in Xcode and examine it there.

```python
mlmodel.save("DistilBert.mlpackage")
```

Note: If the configuration object used returns `True` from `use_legacy_format`, the model can be saved as `ModelName.mlmodel` instead of `.mlpackage`.

#### Exporting a decoder model

Decoder-based models can use a `past_key_values` input that ontains pre-computed hidden-states (key and values in the self-attention blocks), which allows for much faster sequential decoding. This feature is enabled by passing `use_cache=True` to the Transformer model.

To enable this feature with the Core ML exporter, set the `use_past=True` argument when creating the `CoreMLConfig` object:

```python
coreml_config = CTRLCoreMLConfig(base_model.config, task="text-generation", use_past=True)

# or:
coreml_config = CTRLCoreMLConfig.with_past(base_model.config, task="text-generation")
```

This adds multiple new inputs and outputs to the model with names such as `past_key_values_0_key`, `past_key_values_0_value`, ... (inputs) and `present_key_values_0_key`, `present_key_values_0_value`, ... (outputs).

Enabling this option makes the model less convenient to use, since you will have to keep track of many additional tensors, but it does make inference much faster on sequences.

The Transformers model must be loaded with `is_decoder=True`, for example:

```python
base_model = BigBirdForCausalLM.from_pretrained("google/bigbird-roberta-base", torchscript=True, is_decoder=True)
```

TODO: Example of how to use this in Core ML. The `past_key_values` tensors will grow larger over time. The `attention_mask` tensor must have the size of `past_key_values` plus new `input_ids`.

#### Exporting an encoder-decoder model

TODO: properly write this section

You'll need to export the model as two separate Core ML models: the encoder and the decoder.

Export the model like so:

```python
coreml_config = TODOCoreMLConfig(base_model.config, task="text2text-generation", seq2seq="encoder")
encoder_mlmodel = export(preprocessor, base_model.get_encoder(), coreml_config)

coreml_config = TODOCoreMLConfig(base_model.config, task="text2text-generation", seq2seq="decoder")
decoder_mlmodel = export(preprocessor, base_model, coreml_config)
```

When the `seq2seq` option is used, the sequence length in the Core ML model is always unbounded. The `sequence_length` specified in the configuration object is ignored.

This can also be combined with `use_past=True`. TODO: explain how to use this.

#### Validating the model outputs

The final step is to validate that the outputs from the base and exported model agree within some absolute tolerance. You can use the `validate_model_outputs()` function provided by the `exporters.coreml` package as follows.

First enable logging:

```python
from exporters.utils import logging
logger = logging.get_logger("exporters.coreml")
logger.setLevel(logging.INFO)
```

Then validate the model:

```python
from exporters.coreml import validate_model_outputs

validate_model_outputs(
    coreml_config, preprocessor, base_model, mlmodel, coreml_config.atol_for_validation
)
```

Note: `validate_model_outputs` only works on Mac computers, as it depends on the Core ML framework to make predictions with the model.

This function uses the `CoreMLConfig.generate_dummy_inputs()` method to generate inputs for the base and exported model, and the absolute tolerance can be defined in the configuration. We generally find numerical agreement in the 1e-6 to 1e-4 range, although anything smaller than 1e-3 is likely to be OK.

If validation fails with an error such as the following, it doesn't necessarily mean the model is broken:

> ValueError: Output values do not match between reference model and Core ML exported model: Got max absolute difference of: 0.12345

The comparison is done using an absolute difference value, which in this example is 0.12345. That is much larger than the default tolerance value of 1e-4, hence the reported error. However, the magnitude of the activations also matters. For a model whose activations are on the order of 1e+3, a maximum absolute difference of 0.12345 would usually be acceptable.

If validation fails with this error and you're not entirely sure if this is a true problem, call `mlmodel.predict()` on a dummy input tensor and look at the largest absolute magnitude in the output tensor.

### Contributing a new configuration to 🤗 Transformers

We are looking to expand the set of ready-made configurations and welcome contributions from the community! If you would like to contribute your addition to the library, you will need to:

* Implement the Core ML configuration in the `models.py` file
* Include the model architecture and corresponding features in [`~coreml.features.FeatureManager`]
* Add your model architecture to the tests in `test_coreml.py`

### Troubleshooting: What if Core ML Exporters doesn't work for your model?

It's possible that the model you wish to export fails to convert using Core ML Exporters or even when you try to use `coremltools` directly. When running these automated conversion tools, it's quite possible the conversion bails out with an inscrutable error message. Or, the conversion may appear to succeed but the model does not work or produces incorrect outputs.

The most common reasons for conversion errors are:

- You provided incorrect arguments to the converter. The `task` argument should match the chosen model architecture. For example, the `"feature-extraction"` task should only be used with models of type `AutoModel`, not `AutoModelForXYZ`. Additionally, the `seq2seq` argument is required to tell apart encoder-decoder type models from encoder-only or decoder-only models. Passing invalid choices for these arguments may give an error during the conversion process or it may create a model that works but does the wrong thing.

- The model performs an operation that is not supported by Core ML or coremltools. It's also possible coremltools has a bug or can't handle particularly complex models.

If the Core ML export fails due to the latter, you have a couple of options:

1. Implement the missing operator in the `CoreMLConfig`'s `patch_pytorch_ops()` function.

2. Fix the original model. This requires a deep understanding of how the model works and is not trivial. However, sometimes the fix is to hardcode certain values rather than letting PyTorch or TensorFlow calculate them from the shapes of tensors.

3. Fix coremltools. It is sometimes possible to hack coremltools so that it ignores the issue.

4. Forget about automated conversion and [build the model from scratch using MIL](https://coremltools.readme.io/docs/model-intermediate-language). This is the intermediate language that coremltools uses internally to represent models. It's similar in many ways to PyTorch.

5. Submit an issue and we'll see what we can do. 😀

### Known issues

The Core ML exporter writes models in the **mlpackage** format. Unfortunately, for some models the generated ML Program is incorrect, in which case it's recommended to convert the model to the older NeuralNetwork format by setting the configuration object's `use_legacy_format` property to `True`. On certain hardware, the older format may also run more efficiently. If you're not sure which one to use, export the model twice and compare the two versions.

Known models that need to be exported with `use_legacy_format=True` are: GPT2, DistilGPT2.

Using flexible input sequence length with GPT2 or GPT-Neo causes the converter to be extremely slow and allocate over 200 GB of RAM. This is clearly a bug in coremltools or the Core ML framework, as the allocated memory is never used (the computer won't start swapping). After many minutes, the conversion does succeed, but the model may not be 100% correct. Loading the model afterwards takes a very long time and makes similar memory allocations. Likewise for making predictions. While theoretically the conversion succeeds (if you have enough patience), the model is not really usable like this.

## Pushing the model to the Hugging Face Hub

The [Hugging Face Hub](https://huggingface.co) can also host your Core ML models. You can use the [`huggingface_hub` package](https://huggingface.co/docs/huggingface_hub/main/en/index) to upload the converted model to the Hub from Python.

First log in to your Hugging Face account account with the following command:

```bash
huggingface-cli login
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
