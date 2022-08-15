# Hugging Face Transformers DeepSpeed Examples

This repository includes tests for using [DeepSpeed](https://www.deepspeed.ai/) with [transformers](https://huggingface.co/docs/transformers/main/en/main_classes/deepspeed). 
The repository is structured as follows:

* [inference examples](./inference)
* [training examples  _coming soon_](./training)
* [quantization examples _coming soon_](./quantization)

## Getting started

No, matter in which area you are interested in you always will need to setup a correct development/production environment for your use case.
Deepspeed is a framework focused on GPU acceleration and requires a CUDA enabled GPU with a compatible NVIDIA driver. _There is no CPU support._

### Recommened CUDA version

Below you find a list of CUDA versions that are recommended for DeepSpeed.

#### CUDA 11.3

Cuda 11.3 is the recommended version for DeepSpeed, since we have avialble PyTorch versions for it as well.




#### Uninstall current CUDA version

```bash
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" -y
sudo apt-get --purge remove "*nvidia*" -y 
sudo rm -rf /usr/local/cuda*
```



### Default Installation (`PyPi`) _recommended_

```bash
pip install deepspeed
pip install torch
pip install transformers
```

### Installation from scratch

```bash
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.15.0
# DS_BUILD_OPS=1 pip install git+https://github.com/microsoft/DeepSpeed.git

# deepspeed 
sudo apt install libaio-dev
pip install triton==1.0.0
git clone https://github.com/microsoft/DeepSpeed
# mkdir deepspeed/ops/transformer_inference
cd DeepSpeed
DS_BUILD_TRANSFORMER_INFERENCE=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
cd ..
```

run report

```bash
ds_report
```

_expected outcome:_

```bash

```

## Helpful Resources

* [DS Documentation Inference](https://deepspeed.readthedocs.io/en/latest/inference-init.html)
* [Supported Model Architectures](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py)
* [Tutorial: Getting Started with DeepSpeed for Inferencing Transformer based Models](https://www.deepspeed.ai/tutorials/inference-tutorial/)
* [DeepSpeed Launch function for `deepspeed`](https://github.com/microsoft/DeepSpeed/blob/dac9056e13ded1f931171c5f2461761c89fe2595/deepspeed/launcher/launch.py#L90)
* [Interessting Issue for GPT-J](https://github.com/microsoft/DeepSpeed/issues/1332) 
* [T5 Example](https://github.com/microsoft/DeepSpeed/pull/1711/files) 
* [WIP Examples](https://github.com/microsoft/DeepSpeedExamples/tree/inference/General-TP-examples/inference/huggingface)
* [WIP Examples PR](https://github.com/microsoft/DeepSpeedExamples/pull/144)