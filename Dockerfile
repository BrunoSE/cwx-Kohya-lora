# Copyright 2023 Google LLC
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
# Edited by Bruno Stefoni bruno.stefoni@cloudwerx.tech

FROM gcr.io/deeplearning-platform-release/base-gpu.py310

RUN apt-get update

WORKDIR /root

#install sd libraries
RUN git clone https://github.com/Akegarasu/lora-scripts --recurse-submodules --branch main --single-branch

WORKDIR lora-scripts
RUN git branch specific-commit-branch 1f65891c444ad27ed4fb55c5a96a385485bd697e
RUN git switch specific-commit-branch
WORKDIR /root

#install pytorch
RUN pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 xformers==0.0.24 --extra-index-url https://download.pytorch.org/whl/cu118

#install libraries
RUN pip install invisible-watermark>=0.2.0
RUN pip install accelerate==0.25.0
RUN pip install transformers==4.36.2
RUN pip install diffusers[torch]==0.25.0
RUN pip install ftfy==6.1.1
RUN pip install tensorflow==2.10.1 tensorboard==2.10.1
RUN pip install albumentations==1.3.0
RUN pip install opencv-python-headless==4.7.0.72
RUN pip install einops==0.6.1
RUN pip install pytorch-lightning==1.9.0
RUN pip install bitsandbytes==0.39.1
RUN pip install safetensors==0.3.1
RUN pip install gradio==3.16.2
RUN pip install altair==4.2.2
RUN pip install easygui==0.98.3
RUN pip install toml==0.10.2
RUN pip install voluptuous==0.13.1
RUN pip install huggingface-hub==0.20.1
RUN pip install timm==0.6.12
RUN pip install open-clip-torch==2.20.0
RUN pip install cloudml-hypertune==0.1.0.dev6

# Copies the trainer code to the docker image.
COPY train_kohya.py /root/train_kohya.py
COPY hp_train_network.patch /tmp/hp_train_network.patch

WORKDIR lora-scripts/sd-scripts
RUN git apply /tmp/hp_train_network.patch
WORKDIR /root

# Sets up the entry point to invoke the trainer.
#ENTRYPOINT ["python3", "-m", "train_kohya"]
