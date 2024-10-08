"""
Train a diffusion model on images.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import optimizer
from torch.utils.data.dataloader import Sampler
from tqdm import tqdm
import json
import os
import copy

import torch.distributed as dist
import argparse

from guided_diffusion.guided_diffusion import dist_util, logger
from guided_diffusion.guided_diffusion.image_datasets import load_data
from guided_diffusion.guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.guided_diffusion.train_util import TrainLoop
from guided_diffusion.guided_diffusion.unet import AttentionBlock

from peft import LoraConfig, get_peft_model
import pdb

import loralib as lora

def main():


    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

     # Load pre-trained weights into the model (before applying LoRA)
    pretrained_model_path = 'checkpoints/ddpm/64x64_diffusion.pt'
    logger.log(f"Loading pre-trained model from {pretrained_model_path}...")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=dist_util.dev()))

    # adjusted with lora
    model = apply_lora_to_unet_model(model)
    print_all_lora_conv1d_layers(model)

    # This sets requires_grad to False for all parameters without the string "lora_" in their names
    #lora.mark_only_lora_as_trainable(model)

    #lora_model = apply_lora_to_attention_blocks(model, lora_config)

    model.to(dist_util.dev())

    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    
    logger.log("training...")
    
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



def apply_lora_to_attention_block(attention_block, r=16, lora_alpha=32):
     # Clear any existing hooks to avoid unwanted behavior
    if hasattr(attention_block.qkv, '_forward_hooks'):
            attention_block.qkv._forward_hooks.clear()
    # Replace qkv Conv1d with a LoRA-enhanced Conv1d
    if hasattr(attention_block, 'qkv') and isinstance(attention_block.qkv, nn.Conv1d):     
            attention_block.qkv= lora.Conv1d(
            in_channels=attention_block.qkv.in_channels,
            out_channels=attention_block.qkv.out_channels,
            kernel_size=attention_block.qkv.kernel_size[0],  # Assuming a 1D kernel
            r=r,
            lora_alpha=lora_alpha,
            stride=attention_block.qkv.stride,
            padding=attention_block.qkv.padding,
            bias=attention_block.qkv.bias is not None
        )
    if hasattr(attention_block.proj_out, '_forward_hooks'):
            attention_block.proj_out._forward_hooks.clear()
    # Replace proj_out Conv1d with a LoRA-enhanced Conv1d
    if hasattr(attention_block, 'proj_out') and isinstance(attention_block.proj_out, nn.Conv1d):
            attention_block.proj_out = lora.Conv1d(
            in_channels=attention_block.proj_out.in_channels,
            out_channels=attention_block.proj_out.out_channels,
            kernel_size=attention_block.proj_out.kernel_size[0],  # Assuming a 1D kernel
            r=r,
            lora_alpha=lora_alpha,
            stride=attention_block.proj_out.stride,
            padding=attention_block.proj_out.padding,
            bias=attention_block.proj_out.bias is not None
        )

    return attention_block

def apply_lora_to_unet_model(model, r=16, lora_alpha=32):
    # Traverse through the model to find and replace AttentionBlocks
    for name, module in model.named_modules():
        if isinstance(module, AttentionBlock):
            # Apply LoRA to the AttentionBlock and retrieve the modified block
            modified_block = apply_lora_to_attention_block(module, r=r, lora_alpha=lora_alpha)
            # Get the parent module to replace the original block
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model
            if parent_name:  # If the module is nested
                for attr in parent_name.split('.'):
                    parent_module = getattr(parent_module, attr)
            
            # Replace the original AttentionBlock with the modified one
            setattr(parent_module, child_name, modified_block)

    return model

def print_all_lora_conv1d_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, lora.Conv1d):
            print(f"Layer name: {name}")
            print(module)
            print()  # Print an empty line for readability






if __name__ == '__main__':
    main()

