import argparse
import gc
import json
import logging
import time
import os
import torch.nn as nn
import lm_eval
import numpy as np
import torch
from datasets import load_dataset
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from lib.prune_old import calculate_bi, prune_flap, prune_model_blocks



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
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        # default = "google/gemma-2-9b-it",
        help="LLaMA model",
    )  # Huggingface model name
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for sampling the calibration data.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.1, help="Pruning ratio."
    )
    parser.add_argument("--remove_heads", type=int, default=-1, help="Remove num_heads")
    parser.add_argument("--num_blocks", type=int, default=24, help="Total No. of Blocks")
    parser.add_argument(
        "--num_blocks_to_prune", type=int, default=2, help="Remove num blocks"
    )
    parser.add_argument(
        "--pruning_method",
        type=str,
        default="cosine_similarity",
        help="block pruning method",
    )
    parser.add_argument("--pruning_token", type=str, default="all")
    parser.add_argument("--calculate_ppl", type=bool, default=True)
    parser.add_argument(
        "--metrics",
        type=str,
        default="WIFV",
        choices=["IFV", "WIFV", "WIFN", "N/A"],
    )
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["AL-AM"])
    parser.add_argument("--prune_method", type=str, default="flap", choices=["flap"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--save_model",
        type=str,
        default="checkpoints",
        help="Path to save the pruned model.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--block_pruning_bs", type=int, default=2)
    parser.add_argument(
        "--group_size", type=int, default=7, help="Group size, 1 for no GQA."
    )
    parser.add_argument(
        "--gqa_groups", type=int, default=7, help="Group size, 1 for no GQA."
    )
    parser.add_argument(
        "--num_heads", type=int, default=14, help="Number of Query Heads"
    )
    parser.add_argument(
        "--prune_kv_heads",
        type=bool,
        default=True,
        help="Retains KV Heads if set to false",
    )
    parser.add_argument(
        "--start_pruning_layer_idx",
        type=int,
        default=19,
        help="Layer idx post which pruning starts",
    )
    parser.add_argument(
        "--overall_budget",
        type=float,
        default=0.10,
        help="global_budget"
    )
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=896)
    parser.add_argument("--skip_blocks", nargs = '+', type=int, default=[0,1,2,3])
    parser.add_argument("--log_path", type=str, default="prune_1.log")
    parser.add_argument(
        "--strategy",
        type=str,
        default="depth",
        choices=["width_depth", "depth_width","depth","width","baseline"],
    )
    parser.add_argument("--perform_eval", type=bool, default=True)

    args = parser.parse_args()

    logger = logging.getLogger("my_custom_logger")
    logger.setLevel(logging.DEBUG)

    # File handler setup
    file_handler = logging.FileHandler(args.log_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.propagate = False

    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Build the model and tokenizer
    model = get_llm(args.model, args.device)
    if args.device == "auto":
        device = model.hf_device_map["lm_head"]
    else:
        device = torch.device(f"cuda:{args.device}")

    dataset = (
        load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        .filter(lambda example: len(example["text"].split()) > 100)
        .select(list(range(100)))
    )  # 100 samples for pruning metric computation
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.block_pruning_bs, shuffle=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    # tokenizer = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3").tokenizer

    tokenizer.pad_token = tokenizer.eos_token
    orig_params = sum(p.numel() for p in model.parameters()) / 1000**2

    logger.info(f"Unpruned model parameters {orig_params}M")
    if args.strategy == 'depth':

        args.num_blocks_to_prune = round(args.pruning_ratio * args.num_blocks)
        bi_scores = calculate_bi(
            model,
            dataloader,
            tokenizer,
            args.pruning_method,
            args.pruning_token,
        )
        block_pruned_model = prune_model_blocks(
            model, bi_scores, args.num_blocks_to_prune, args.skip_blocks
        )
        block_pruned_params = (
            sum(p.numel() for p in block_pruned_model.parameters()) / 1000**2
        )
        logger.info(
            f"Compression After Block Pruning {1 - block_pruned_params/orig_params}"
        )
        compression = {"compression_ratio" : 1 - block_pruned_params/orig_params}
        del model
        torch.cuda.empty_cache()
        gc.collect()
        # block_pruned_model.to(device)
        model = block_pruned_model
        print(model)

    
    elif args.strategy == "width":
        prune_flap(args, model, tokenizer, device)
        width_pruned_params = sum(p.numel() for p in model.parameters()) / 1000**2
        logger.info(
            f"Compression After Width Pruning {1 - width_pruned_params/orig_params}"
        )
        compression = {"compression_ratio" : 1 - width_pruned_params/orig_params}
        print(compression)

    

    model = model.half()
    # Evaluate the model
    if args.eval:
        lm_obj = HFLM(pretrained=model, batch_size="auto")
        task_manager = lm_eval.tasks.TaskManager()
        results = []
        result = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=["boolq","wikitext"],      
            task_manager=task_manager,
        )

if __name__ == "__main__":
    main()