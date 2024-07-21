"""
Generate a large batch of scRNA-seq gene expression samples from a model and save them as a large
numpy array. 
"""
import argparse
import time
import os

import numpy as np
import torch as th
import torch.distributed as dist
import random

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
)

# os.environ["CUDA_VISIBLE_DEVICES"]='0'

def save_data(all_cells, data_dir):
    np.save(data_dir, all_cells)
    return

def main(cell_type, args_dict):
    setup_seed(1234)

    dist_util.setup_dist()
    # logger.configure(dir='checkpoint/sample_logs')

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_dict
    )
    model.load_state_dict(
        dist_util.load_state_dict(args_dict['model_path'], map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    print("sampling...")
    all_cells = []
    
    if len(cell_type) > 1:
        y = th.tensor([cell_type] * args_dict['batch_size'])
    else:
        y = th.tensor(cell_type * args_dict['batch_size'])
    
    # print(f'cell_type:{len(cell_type)}\ny:{y.shape}')
    
    elapse_all = 0.
    while len(all_cells) * args_dict['batch_size'] < args_dict['num_samples']:
        
        sample_fn = (
            diffusion.p_sample_loop if not args_dict['use_ddim'] else diffusion.ddim_sample_loop
        )
        
        start = time.process_time() 
        sample, _ = sample_fn(
            model,
            (args_dict['batch_size'], args_dict['input_dim']), 
            clip_denoised=args_dict['clip_denoised'],
            y=y,
            start_time=diffusion.betas.shape[0],
        )
        elapse_all = elapse_all + time.process_time() - start

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_cells.extend([sample.cpu().numpy() for sample in gathered_samples])
        print(f"created {len(all_cells) * args_dict['batch_size']} samples")

    arr = np.concatenate(all_cells, axis=0)
    os.makedirs(args_dict['sample_dir'], exist_ok=True)
    save_data(arr, f"{args_dict['sample_dir']}/cell{''.join(map(str, cell_type))}_cache{args_dict['cache_interval']}_{'non_uniform' if args_dict['non_uniform'] else 'uniform'}")

    dist.barrier()
    print("sampling complete")
    return elapse_all


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/workplace/scDiffusion_full/checkpoint/pbmc68k_diffusion_based_on_vae800000/model800000.pt")
    parser.add_argument("--sample_dir", type=str, default="/home/workplace/scDiffusion_full/generation/pbmc68k/")
    
    # parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--num_classes", nargs='+', type=int, default=11)
    parser.add_argument("--branch", type=int, default=0)
    parser.add_argument("--cache_interval", type=int, default=5)
    parser.add_argument("--non_uniform", type=bool, default=False)
    parser.add_argument("--clip_denoised", type=bool, default=False)
    parser.add_argument("--use_ddim", type=bool, default=False)
    parser.add_argument("--num_samples", type=int, default=9000)
    parser.add_argument("--batch_size", type=int, default=3000)
    
    # add_dict_to_argparser(parser, model_and_diffusion_defaults())
    args, _ = parser.parse_known_args()
    args_dict = {arg: getattr(args, arg) for arg in vars(args)}
    args_dict.update(model_and_diffusion_defaults())
    return args_dict

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True # 设置随机数种子


if __name__ == "__main__":
    args_dict = create_argparser()
    total = 0.
    ls = []
    
    ### one condition
    for i in range(args_dict['num_classes'][0]):
        elapse = main([i], args_dict)
        ls.append(elapse)
        total = total + elapse
        print(f'-----------time elpse:{elapse}')
    
    ## multi-condition
    # for i in range(args_dict['num_classes'][0]):
    #     for j in range(args_dict['num_classes'][1]):
    #         elapse = main([i, j], args_dict)
    #         ls.append(elapse)
    #         total = total + elapse
    #         print(f'-----------time elpse:{elapse}')
    
    print(ls)
    print(total)
