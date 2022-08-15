# Hugging Face Transformers DeepSpeed Examples

This repository includes tests for using [DeepSpeed](https://www.deepspeed.ai/) with [transformers](https://huggingface.co/docs/transformers/main/en/main_classes/deepspeed). 
The repository is structured as follows:

* [inference examples](./inference)
* [training examples  _coming soon_](./training)
* [quantization examples _coming soon_](./quantization)

## Getting started

No, matter in which area you are interested in you always will need to setup a correct development/production environment for your use case.
Deepspeed is a framework focused on GPU acceleration and requires a CUDA enabled GPU with a compatible NVIDIA driver. _There is no CPU support._


### Default Installation (`PyPi`) _recommended_


```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers[sentencepiece] == 4.21.1
pip install deepspeed
pip install datasets evaluate
```

### Docker container

The repository contains also a `Dockerfile` with all steps for building a containzerized container environment.

* [Dockerfile](./Dockerfile)

### Installation from scratch

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers[sentencepiece] == 4.21.1

# deepspeed 
sudo apt install libaio-dev
pip install triton==1.0.0
git clone https://github.com/microsoft/DeepSpeed
# mkdir deepspeed/ops/transformer_inference
cd DeepSpeed
pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
cd ..
```

### Validation installation 

run report

```bash
ds_report
```

_expected outcome:_

```bash
```



### Recommened CUDA version

Below you find a list of CUDA versions that are recommended for DeepSpeed. 

#### AWS Deep Learning AMIs

If you are running on AWS you can also directly select a Deep Learning AMI with the correct CUDA version installed. You can find a list of available DLAMI [here](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html#appendix-ami-release-notes-gpu).


#### CUDA 11.3

Cuda 11.3 is the recommended version for DeepSpeed, since we have avialble PyTorch versions for it as well. [Reference Gist](https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73).

If you have a different CUDA version installed remove it first.

```bash
# system update
sudo apt-get update
sudo apt-get upgrade -y

# install other import packages
sudo apt-get install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# install nvidia driver with dependencies
sudo apt install -y libnvidia-common-470 libnvidia-gl-470 nvidia-driver-470

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

 # installing CUDA-11.3
sudo apt install -y cuda-11-3 

# setup your paths
echo 'export PATH=/usr/local/cuda-11.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# install cuDNN v11.3
# First register here: https://developer.nvidia.com/developer-program/signup

CUDNN_TAR_FILE="cudnn-11.3-linux-x64-v8.2.1.32.tgz"
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz
tar -xzvf ${CUDNN_TAR_FILE}

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.3/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64/
sudo chmod a+r /usr/local/cuda-11.3/lib64/libcudnn*

# Finally, to verify the installation, check
nvidia-smi
nvcc -V
```

#### CUDA 11.6 

Similar to [CUDA 11.3](#cuda-11-3) we can install CUDA 11.6. Repalce the version with `11-6` in the above command.

#### Uninstall current CUDA version

```bash
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" -y
sudo apt-get --purge remove "*nvidia*" -y 
sudo apt-get autoremove -y
sudo apt-get autoclean -y
sudo rm -rf /usr/local/cuda*
sudo rm /etc/apt/sources.list.d/cuda*
# reboots system
sudo reboot
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