"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
# from guided_diffusion.cell_datasets import load_data
# from guided_diffusion.cell_datasets_WOT import load_data
from guided_diffusion.cell_datasets_sapiens import load_data
# from guided_diffusion.cell_datasets_muris import load_data
# from guided_diffusion.cell_datasets_pbmc import load_data
# from guided_diffusion.cell_datasets_lung import load_data
# from guided_diffusion.cell_datasets_Alles import load_data
# from guided_diffusion.cell_datasets_muris_mam_spl_T_B import load_data
# from guided_diffusion.cell_datasets_Baron_human import load_data
# from guided_diffusion.cell_datasets_ALIGNED_Mus_musculus_Mammary_Gland import load_data

from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    # args_to_dict,
    # add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

import torch
import numpy as np
import random

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='-1'

def main():
    setup_seed(1234)
    args_dict = create_argparser()

    dist_util.setup_dist()
    # logger.configure(dir='../output/logs/'+args.model_name)  # log file

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_dict)
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args_dict['schedule_sampler'], diffusion)

    print("creating data loader...")
    data = load_data(
        data_dir=args_dict['data_dir'],
        batch_size=args_dict['batch_size'],
        vae_path=args_dict['vae_path'],
        train_vae=False,
    )

    print("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args_dict['batch_size'],
        microbatch=args_dict['microbatch'],
        lr=args_dict['lr'],
        ema_rate=args_dict['ema_rate'],
        log_interval=args_dict['log_interval'],
        save_interval=args_dict['save_interval'],
        resume_checkpoint=args_dict['resume_checkpoint'],
        use_fp16=args_dict['use_fp16'],
        fp16_scale_growth=args_dict['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args_dict['weight_decay'],
        lr_anneal_steps=args_dict['lr_anneal_steps'],
        model_name=args_dict['model_name'],
        save_dir=args_dict['save_dir']
    ).run_loop()


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=800001, #500000
        batch_size=128, #128
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10000000,
        save_interval=400000,#200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        
        # ##muris
        # data_dir="/home/zqzhao/workplace/scDiffusion-main/dataset/tabular_muris/tabular_muris.h5ad",
        # vae_path = '/home/zqzhao/workplace/scDiffusionV2/checkpoint/VAE/model_seed=0_step=150000.pt',
        # model_name="muris_diffusion_based_on_vae150000",
        # save_dir='/home/zqzhao/workplace/scDiffusion_full/checkpoint',
        # num_classes=12,
        
        ##pbmc68k
        # data_dir="/home/zqzhao/workplace/scDiffusion-main/dataset/pbmc68k/filtered_matrices_mex/hg19/",
        # vae_path = '/home/zqzhao/workplace/scDiffusion_full/checkpoint/VAE/pbmc68k/model_seed=0_step=800000.pt',
        # model_name="pbmc68k_diffusion_based_on_vae800000",
        # save_dir='/home/zqzhao/workplace/scDiffusion_full/checkpoint',
    )
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--num_classes", nargs='+', type=int, default=11)
    parser.add_argument("--branch", type=int, default=0)
    parser.add_argument("--cache_interval", type=int, default=5)
    parser.add_argument("--non_uniform", type=bool, default=False)
    parser.add_argument("--data_dir", type=str, default='/home/zqzhao/workplace/scDiffusion-main/dataset/pbmc68k/filtered_matrices_mex/hg19/')
    parser.add_argument("--vae_path", type=str, default='/home/zqzhao/workplace/scDiffusion_full/checkpoint/VAE/pbmc68k/model_seed=0_step=800000.pt')
    parser.add_argument("--save_dir", type=str, default='/home/zqzhao/workplace/scDiffusion_full/checkpoint')
    parser.add_argument("--model_name", type=str, default='pbmc68k_diffusion_based_on_vae800000')
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    args_dict = {arg: getattr(args, arg) for arg in vars(args)}
    
    defaults.update(model_and_diffusion_defaults())
    args_dict.update(defaults)
    # add_dict_to_argparser(parser, defaults)
    
    return args_dict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
