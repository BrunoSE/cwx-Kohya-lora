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

import subprocess
import os
import argparse
import re
import torch
from safetensors.torch import save_file
import json
from datetime import datetime
from pytz import timezone
from diffusers import AutoPipelineForText2Image


def main(args):
    # make folder per experiment
    EXPERIMENT_TAG = args.experiment_tag
    OUTPUT_DIR = args.output_storage
    if OUTPUT_DIR.endswith("/"):
        OUTPUT_DIR = OUTPUT_DIR[:-1]

    pst_timezone = timezone('US/Pacific')
    # Get current time in PST
    pst_time = datetime.now(pst_timezone)
    TIMESTAMP = pst_time.strftime("%Y%m%d%H%M%S")

    # Construct the full folder path with experiment tag
    if EXPERIMENT_TAG:
        OUTPUT_DIR = f"{OUTPUT_DIR}/{TIMESTAMP}_{EXPERIMENT_TAG}"
    else:
        OUTPUT_DIR = f"{OUTPUT_DIR}/{TIMESTAMP}"

    # gsutil needs another format
    if OUTPUT_DIR.startswith('/gcs/'):
        OUTPUT_DIR_gs = OUTPUT_DIR.replace('/gcs/','gs://',1)

    # create new empty folder in bucket
    subprocess.run(f"gsutil cp -p /dev/null {OUTPUT_DIR_gs}/", shell=True)

    # save to json
    with open(f"{OUTPUT_DIR}/parameters.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # config accelerate for kohya
    subprocess.run("accelerate config default", shell=True)
    subprocess.run("cat /root/.cache/huggingface/accelerate/default_config.yaml", shell=True)

    # assign parsed args to variables
    METHOD = args.method
    NUM_CPU_THREADS = int(args.num_cpu_threads)
    MODEL_NAME = args.pretrained_model_name_or_path
    INSTANCE_DIR= args.input_storage
    METADATA_DIR = args.metadata_storage
    DISPLAY_NAME = args.display_name
    RESOLUTION = args.resolution
    MAX_EPOCHS = int(args.max_train_epochs)
    LR = float(args.lr)
    UNET_LR = float(args.unet_lr)
    TEXT_ENCODER_LR = float(args.text_encoder_lr)
    LR_SCHEDULER = args.lr_scheduler
    NETWORK_DIM = int(args.network_dim)
    NETWORK_ALPHA = int(args.network_alpha)
    BATCH_SIZE = int(args.batch_size)
    SAVE_N_EPOCHS = int(args.save_every_n_epochs)
    NETWORK_WEIGHTS = args.network_weights
    REG_DIR = args.reg_dir
    USE_8BIT_ADAM = bool(args.use_8bit_adam)
    USE_LION = bool(args.use_lion)
    NOISE_OFFSET = int(args.noise_offset)
    HPO = args.hpo

    # Create variables from parsed arguments
    NETWORK_MODULE = args.network_module
    LR_WARMUP_STEPS = int(args.lr_warmup_steps)  # Convert to int
    LR_SCHEDULER_NUM_CYCLES = int(args.lr_scheduler_num_cycles)  # Convert to int
    MIXED_PRECISION = args.mixed_precision
    SAVE_PRECISION = args.save_precision
    SEED = int(args.seed)  # Convert to int
    CACHE_LATENTS = bool(args.cache_latents)  # Convert to bool
    CLIP_SKIP = int(args.clip_skip)  # Convert to int
    PRIOR_LOSS_WEIGHT = float(args.prior_loss_weight)  # Convert to float
    MAX_TOKEN_LENGTH = int(args.max_token_length)  # Convert to int
    CAPTION_EXTENSION = args.caption_extension
    SAVE_MODEL_AS = args.save_model_as
    MIN_BUCKET_RESO = int(args.min_bucket_reso)  # Convert to int
    MAX_BUCKET_RESO = int(args.max_bucket_reso)  # Convert to int
    KEEP_TOKENS = int(args.keep_tokens)  # Convert to int
    XFORMERS = bool(args.xformers)  # Convert to bool
    NO_HALF_VAE = bool(args.no_half_vae)  # Convert to bool
    OPTIMIZER_ARGS = args.optimizer_args
    MAX_DATA_LOADER_N_WORKERS = int(args.max_data_loader_n_workers)  # Convert to int
    NETWORK_TRAIN_UNET_ONLY = bool(args.network_train_unet_only)  # Convert to bool
    FULL_BF16 = bool(args.full_bf16)  # Convert to bool

    OPTIMIZER_TYPE = args.optimizer_type
    ENABLE_BUCKET = args.enable_bucket

    PROMPT_1 = args.my_prompt_1

    if METHOD == "kohya_lora":
        os.chdir("/root/lora-scripts")
        # for complex commands, with many args, use string + `shell=True`:
        cmd_str = (f'accelerate launch --num_cpu_threads_per_process={NUM_CPU_THREADS} sd-scripts/sdxl_train_network.py '
                   f'--pretrained_model_name_or_path="{MODEL_NAME}" '
                   f'--train_data_dir="{INSTANCE_DIR}" '
                   f'--output_dir="{OUTPUT_DIR}" '
                   f'--logging_dir="{OUTPUT_DIR}/logs" '
                   f'--log_prefix="{DISPLAY_NAME}_logs" '
                   f'--resolution="{RESOLUTION}" '
                   f'--network_module="{NETWORK_MODULE}" '
                   f'--max_train_epochs={MAX_EPOCHS} '
                   f'--learning_rate={LR} '
                   f'--unet_lr={UNET_LR} '
                   f'--text_encoder_lr={TEXT_ENCODER_LR} '
                   f'--lr_scheduler="{LR_SCHEDULER}" '
                   f'--lr_warmup_steps={LR_WARMUP_STEPS} '
                   f'--lr_scheduler_num_cycles={LR_SCHEDULER_NUM_CYCLES} '
                   f'--network_dim={NETWORK_DIM} '
                   f'--network_alpha={NETWORK_ALPHA} '
                   f'--output_name="{DISPLAY_NAME}" '
                   f'--train_batch_size={BATCH_SIZE} '
                   f'--save_every_n_epochs={SAVE_N_EPOCHS} '
                    f"--mixed_precision={MIXED_PRECISION} "
                    f"--save_precision={SAVE_PRECISION} "
                    f"--seed={SEED} "
                    f"--clip_skip={CLIP_SKIP} "
                    f"--prior_loss_weight={PRIOR_LOSS_WEIGHT} "
                    f"--max_token_length={MAX_TOKEN_LENGTH} "
                    f"--caption_extension={CAPTION_EXTENSION} "
                    f"--save_model_as={SAVE_MODEL_AS} "
                    f"--min_bucket_reso={MIN_BUCKET_RESO} "
                    f"--max_bucket_reso={MAX_BUCKET_RESO} "
                    f"--keep_tokens={KEEP_TOKENS} "
                    f"--max_data_loader_n_workers={MAX_DATA_LOADER_N_WORKERS} "
                    f"--optimizer_type {OPTIMIZER_TYPE} "
                    f"--optimizer_args {OPTIMIZER_ARGS} "
                    f'--hpo="{HPO}"')

        if ENABLE_BUCKET:
            cmd_str += f' --enable_bucket'
        if CACHE_LATENTS:
            cmd_str += f' --cache_latents' 
        if XFORMERS:
            cmd_str += f' --xformers' 
        if NO_HALF_VAE:
            cmd_str += f' --no_half_vae'
        if NETWORK_TRAIN_UNET_ONLY:
            cmd_str += f' --network_train_unet_only'
        if FULL_BF16:
            cmd_str += f' --full_bf16'
        if NETWORK_WEIGHTS:
            cmd_str += f' --network_weights="{NETWORK_WEIGHTS}"'
        if REG_DIR:
            cmd_str += f' --reg_data_dir="{REG_DIR}"'

        if USE_8BIT_ADAM:
            if OPTIMIZER_TYPE is None:
                print("Warning: ignoring --use_8bit_adam because --optimizer_type is not set to Adam")
            elif OPTIMIZER_TYPE!="Adam":
                print("Warning: ignoring --use_8bit_adam because --optimizer_type is not set to Adam")
            else:
                cmd_str += f' --use_8bit_adam'
            
        if USE_LION:
            if OPTIMIZER_TYPE:
                print("Warning: ignoring --use_lion because --optimizer_type was used")
            else:
                cmd_str += f' --use_lion_optimizer'

        if NOISE_OFFSET:
            cmd_str += f' --noise_offset={NOISE_OFFSET}'
        if METADATA_DIR is not None:
            cmd_str += f' --in_json="{METADATA_DIR}"'
        # add --shuffle_caption ???
    
    # start training
    subprocess.run(cmd_str, shell=True)

    # files with ".safetensors" extension
    model_files = [file for file in os.listdir(OUTPUT_DIR) if file.endswith(".safetensors")]
    print(f"Found model files: {model_files}")


    if PROMPT_1:
        for model_file in model_files:
            print(f"Generating images with model {model_file} for test inspection..")
            pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
            pipeline.load_lora_weights(f'{OUTPUT_DIR}/{model_file}', weight_name=model_file)

            # go over prompts
            model_file_ = model_file.replace(".safetensors",'')
            image1 = pipeline(PROMPT_1).images[0]
            print("Saving images..")
            image1.save(f"{OUTPUT_DIR}/{model_file_}_img_1.png")

            pipeline = None

    # idea: save image outputs to nfs for easier inspection ?

    if bool(args.save_nfs) == True:
        nfs_path = args.nfs_mnt_dir

        if not os.path.exists(nfs_path):
            print("nfs not exist")
        else:
            if not os.path.exists(nfs_path + '/kohya'):
               os.mkdir(nfs_path + '/kohya')
               print(f"{nfs_path}/kohya has been created.")
            else:
               print(f"{nfs_path}/kohya already exists.")
            copy_cmd = f'cp {OUTPUT_DIR}/*.safetensors {nfs_path}/kohya'
            subprocess.run(copy_cmd, shell=True)
            subprocess.run(f'ls {nfs_path}/kohya', shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, default="", help="a custom tag to help track experiments")

    parser.add_argument("--method", type=str, default="kohya_lora", help="keep as kohya_lora")
    parser.add_argument("--num_cpu_threads", type=int, default=12, help="num of cpu threads per process")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="bucket_name/model_folder")
    parser.add_argument("--input_storage", type=str,default="/root/dog_image_resize", help="/gcs/bucket_name/input_image_folder")
    parser.add_argument("--metadata_storage", type=str, default=None, help="metadata json path, for native training")
    parser.add_argument("--output_storage", type=str, default="/root/dog_output", help="/gcs/bucket_name/output_folder")
    parser.add_argument("--display_name", type=str, default="ydz", help="prompt")
    parser.add_argument("--resolution", type=str, default="768,768", help="resolution group")
    parser.add_argument("--max_train_epochs", type=int, default=4, help="max train epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--unet_lr", type=float, default=1e-4, help="unet learning rate")
    parser.add_argument("--text_encoder_lr", type=float, default=1e-5, help="text encoder learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="")
    parser.add_argument("--network_dim", type=int, default=32, help="network dim 4~128")
    parser.add_argument("--network_alpha", type=int, default=32, help="often=network dim")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--save_every_n_epochs", type=int, default=2, help="save every n epochs")
    parser.add_argument("--network_weights", type=str, default="", help="lora model path,/gcs/bucket_name/lora_model")
    parser.add_argument("--reg_dir", type=str, default="", help="regularization data path")
    parser.add_argument("--use_8bit_adam", type=bool, default=False, help="use 8bit adam optimizer")
    parser.add_argument("--use_lion", type=bool, default=False, help="lion optimizer")
    parser.add_argument("--noise_offset", type=int, default=0, help="0.1 if use")
    parser.add_argument("--save_nfs", type=bool, default=False, help="if save the model to file store")
    parser.add_argument("--save_nfs_only", type=bool, default=False, help="only copy file from gcs to filestore, no training")
    parser.add_argument("--nfs_mnt_dir", type=str, default="/mnt/nfs/model_repo", help="Filestore's mount directory")
    parser.add_argument("--hpo", type=str, default="n", help="if using hyper parameter tuning")

    parser.add_argument("--network_module", type=str, default="networks.lora", help="network module to use")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="number of learning rate warmup steps")
    parser.add_argument("--lr_scheduler_num_cycles", type=int, default=1, help="number of learning rate scheduler cycles")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="mixed precision mode to use")
    parser.add_argument("--save_precision", type=str, default="fp16", help="precision for saving the model")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--cache_latents", type=bool, default=True, help="whether to cache latents")
    parser.add_argument("--clip_skip", type=int, default=0, help="number of layers to skip, 0 means none")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="weight for the prior loss")
    parser.add_argument("--max_token_length", type=int, default=225, help="maximum token length")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="caption file extension")
    parser.add_argument("--save_model_as", type=str, default="safetensors", help="format to save the model")
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum bucket resolution")
    parser.add_argument("--max_bucket_reso", type=int, default=2028, help="maximum bucket resolution")
    parser.add_argument("--keep_tokens", type=int, default=0, help="number of tokens to keep")
    parser.add_argument("--xformers", type=bool, default=True, help="whether to use transformers")
    parser.add_argument("--no_half_vae", type=bool, default=True, help="whether to disable half VAE")
    parser.add_argument("--optimizer_args", type=str, default="", help="additional optimizer arguments")  # Consider clarifying this help message
    parser.add_argument("--max_data_loader_n_workers", type=int, default=0, help="maximum number of data loader workers")
    parser.add_argument("--network_train_unet_only", type=bool, default=True, help="whether to train only the U-Net part of the network")
    parser.add_argument("--full_bf16", type=bool, default=True, help="whether to use full BF16 precision")

    parser.add_argument("--optimizer_type", type=str, default="Adafactor", help="optimizer selection")
    parser.add_argument("--enable_bucket", type=bool, default=True, help="bucketing option")

    parser.add_argument("--my_prompt_1", type=str, default="", help="a custom prompt to generate a test image")
    # test promps could be better handled by having a testprompts.json file in a bucket folder in GCS

    # others parameters not yet added to the code
    #    --lowram 
    #    --bucket_reso_steps=64 
    #    --min_snr_gamma=5
    #    --gradient_checkpointing

    args = parser.parse_args()
    print(args)
    if bool(args.save_nfs_only) == True:
        nfs_path = args.nfs_mnt_dir #"/mnt/nfs/model_repo"
        if not os.path.exists(nfs_path):
            print("nfs not exist")
        else:
            if not os.path.exists(nfs_path + '/kohya'):
               os.mkdir(nfs_path + '/kohya')
               print(f"{nfs_path}/kohya has been created.")
            else:
               print(f"{nfs_path}/kohya already exists.")
            copy_cmd = f'cp {args.output_storage}/*.safetensors {nfs_path}/kohya'
            subprocess.run(copy_cmd, shell=True)
            subprocess.run(f'ls {nfs_path}/kohya', shell=True)  
    else:
       main(args)

