import argparse
import os 
import numpy as np
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.hf_llama.modeling_llama import LlamaForCausalLM
from datasets import load_dataset
from importlib.metadata import version
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from lib.prune import prune_flap, calculate_bi, prune_model_blocks
from lib.eval import eval_ppl
import lm_eval
from lm_eval.models.huggingface import HFLM
import sys

def get_llm(model, device):
    if device=='auto':
        model = AutoModelForCausalLM.from_pretrained(
            model, 
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(torch.device(f"cuda:{device}"))
    
    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros(model.model.layers[i].self_attn.o_proj.weight.shape[0], device=model.model.layers[i].self_attn.o_proj.weight.device, dtype=torch.float16))  # 或 'cuda'
        model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(torch.zeros(model.model.layers[i].mlp.down_proj.weight.shape[0], device=model.model.layers[i].mlp.down_proj.weight.device, dtype=torch.float16))  # 或 'cuda'
        torch.nn.init.zeros_(model.model.layers[i].self_attn.o_proj.bias)
        torch.nn.init.zeros_(model.model.layers[i].mlp.down_proj.bias)
        
    model.seqlen = 128
    return model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3-8B", help='LLaMA model')    # Huggingface model name
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=1024, help='Number of calibration samples.')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='Pruning ratio.')
    parser.add_argument('--remove_heads', type=int, default=-1, help='Remove num_heads')
    parser.add_argument('--num_blocks_to_prune', type=int, default=3, help='Remove num blocks')
    parser.add_argument('--pruning_method', type=str, default="cosine_similarity", help='block pruning method')
    parser.add_argument('--pruning_token', type=str, default="all")
    parser.add_argument('--calculate_ppl', type=bool, default=True)
    parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["AL-AM"])
    parser.add_argument("--prune_method", type=str, default="flap", choices=["flap"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--block_pruning_bs', type=int, default=2)
    parser.add_argument('--gqa_groups', type=int, default=7, help='Number of gqa groups, 1 for no GQA.')
    parser.add_argument('--start_pruning_layer_idx', type=int, default=18, help='Layer idx post which pruning starts')
    parser.add_argument('--head_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=896)
    parser.add_argument('--skip_blocks', type=list, default=[1])
    parser.add_argument('--block_first', type=bool, default=-True)
    parser.add_argument('--perform_eval', type=bool, default=-True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.device=='auto':
        device = model.hf_device_map["lm_head"]
    else:
        device = torch.device(f"cuda:{args.device}")
    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.device)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test").filter(lambda example: len(example["text"].split())>100).select(list(range(100))) # 100 samples for pruning metric computation
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.block_pruning_bs, shuffle=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    print(f"Unpruned model parameters {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.2f}B")

    if args.block_first:
        print(f"Performing Block Pruning followed by FLAP")
        bi_scores = calculate_bi(model, dataloader, tokenizer, args.pruning_method, args.pruning_token)
        block_pruned_model = prune_model_blocks(model, bi_scores, args.num_blocks_to_prune, args.skip_blocks)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        block_pruned_model.to(device)
        prune_flap(args, block_pruned_model, tokenizer, device)
        print(f"Pruned model parameter {sum(p.numel() for p in block_pruned_model.parameters()) / 1000 ** 3:.2f}B")

    else:
        print("Performing FLAP followed by Block Pruning")
        prune_flap(args, model, tokenizer, device)
        bi_scores = calculate_bi(model, dataloader, tokenizer, args.pruning_method, args.pruning_token)
        block_pruned_model = prune_model_blocks(model, bi_scores, args.num_blocks_to_prune, args.skip_blocks)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        block_pruned_model.to(device)

if __name__ == '__main__':
    main()
