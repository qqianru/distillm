import time
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import RandomSampler
from torch.optim import AdamW
import deepspeed

import random
import json
from tqdm import tqdm
import math
import datetime

import distillm
print(dir(distillm))  # Check if forward_kl is listed

os.environ["PYTHONPATH"] = "/content/distillm:" + os.environ.get("PYTHONPATH", "")


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig)

from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from distillm import forward_kl, reverse_kl, js_distance, tv_distance
from distillm import skewed_forward_kl, skewed_reverse_kl
from distillm import SampleGenerator, ReplayBuffer


from rouge_metric import compute_metrics

from peft import PeftModel

torch.set_num_threads(4)


def get_teacher_model(args, device):
    print("teacher_model_path",args.teacher_model_path)
    config = AutoConfig.from_pretrained(args.teacher_model_path)   
    if args.model_parallel:
        raise NotImplementedError
    else:
        config.is_model_parallel = False
        try: model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float16)
        except:
            model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config, device_map={"": device}, torch_dtype=torch.float32)
            model = model.half()
        
        if args.peft is not None and args.teacher_peft_path is not None:
            if args.peft == "lora":
                model = PeftModel.from_pretrained(model, args.teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

    model.eval()
    
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


def prepare_dataset(args, tokenizer):
    data = {}
    rng_sample = random.Random(args.seed)
    if args.do_train:
        data["train"] = LMTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, args.data_dir, "valid", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
        
    # pre-trained dataset
    if args.do_train and args.lm_data_dir is not None:
        data["pt_train"] = LMTrainDataset(args, tokenizer, args.lm_data_dir, "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["pt_train"]))
    return data


def pt_loss(args, model, model_batch, no_model_batch):
    loss_mask = (no_model_batch["label"] != -100).int()
    outputs = model(**model_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), no_model_batch["label"].view(-1))
    return lm_loss


def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
    if args.model_parallel:
        raise NotImplementedError
    else:
        if "sfkl" in args.type:
            distil_loss = skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "srkl" in args.type:
            distil_loss = skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=args.skew_alpha)
        elif "jsd" in args.type:
            distil_loss = js_distance(logits, teacher_logits, no_model_batch)
        elif "tvd" in args.type:
            distil_loss = tv_distance(logits, teacher_logits, no_model_batch)
        elif "fkl" in args.type or args.type == "kd":
            distil_loss = forward_kl(logits, teacher_logits, no_model_batch)
        elif "rkl" in args.type:
            distil_loss = reverse_kl(logits, teacher_logits, no_model_batch)
        else:
            raise NotImplementedError
    return distil_loss


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences
    
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss


def finetune(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device, teacher_model=None):
    print_rank("Start Fine-tuning")

    # Check if distributed is initialized
    dist_initialized = dist.is_initialized()
    if dist_initialized:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
    else:
        dp_world_size = 1
        dp_rank = 0
    dp_group = None
    loss_func = nn.CrossEntropyLoss()

    # Debug print first batch before dataloader
    for batch in dataset["train"]:
        print("Batch structure-train_dataloader-before-dataloader:", batch)
        break

    if dp_world_size > 1:
        sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    else:
        sampler = RandomSampler(dataset["train"])  # or no sampler at all

    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate
    )

    for batch in train_dataloader:
        print("Batch structure-train_dataloader:", batch)
        break

    if "pt_train" in dataset:
        if dp_world_size > 1:
            pt_sampler = DistributedSampler(dataset["pt_train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
        else:
            pt_sampler = RandomSampler(dataset["pt_train"])
        
        pt_train_dataloader = DataLoader(
            dataset['pt_train'], sampler=pt_sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["pt_train"].collate
        )
        pt_train_iter = iter(pt_train_dataloader)

    student_generator = SampleGenerator(args, tokenizer)
    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0

    adaptive_threshold = args.init_threshold if "adaptive" in args.type else None
    for batch in dataset["dev"]:
        print("Batch dataset_dev:", batch)
        break
    
    prev_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", 0, device, adaptive_threshold)
    replay_buffer = ReplayBuffer(args)

    for epoch in range(args.epochs):
        if dp_world_size > 1:
            sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)

            if args.lm_data_dir is not None:
                try:
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                except StopIteration:
                    pt_train_iter = iter(pt_train_dataloader)
                    pt_model_batch, pt_no_model_batch, pt_gen_data = next(pt_train_iter)
                    
                dataset["pt_train"].move_to_device(pt_model_batch, pt_no_model_batch, pt_gen_data, device)

            torch.cuda.synchronize()
            st_time = time.time()

            # sampling ratio for adaptive strategies
            samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters) if "adaptive" in args.type else None
            if "adaptive" in args.type:
                if args.replay_ratio == "constant":
                    samp_threshold = adaptive_threshold * 0.5
                elif args.replay_ratio == "increasing":
                    samp_threshold = adaptive_threshold * global_step / args.total_iters
                else:
                    samp_threshold = adaptive_threshold * (1 - global_step / args.total_iters)

            # data generation
            if args.student_gen:
                r = np.random.uniform(0, 1)
                if "mixed" in args.type and r < args.mixed_alpha:
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)
                    
                elif "adaptive" in args.type and (r < samp_threshold or (r < adaptive_threshold and len(replay_buffer) < args.capacity)):
                    model_batch = student_generator.run_sample(model, gen_data)
                    no_model_batch["label"] = model_batch.pop("no_model_batch")
                    if args.model_type in ["opt"]:
                        model_batch.pop('position_ids', None)
                    
                    replay_buffer.move_to_memory(model_batch, no_model_batch)
                    
                elif "adaptive" in args.type and r < adaptive_threshold:
                    model_batch, no_model_batch = replay_buffer.sample()
                    model_batch, no_model_batch = replay_buffer.move_to_device(model_batch, no_model_batch, device)
                    
                model.train()

            outputs = model(**model_batch, use_cache=False)
            logits = outputs.logits
            lm_loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            if teacher_model is not None:
                distil_loss = get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits)
                loss = (1 - args.kd_ratio) * lm_loss + args.kd_ratio * distil_loss
            else:
                loss = lm_loss

            if args.lm_data_dir is not None:
                assert args.lm_coef is not None
                loss += args.lm_coef * pt_loss(args, model, pt_model_batch, pt_no_model_batch)

            model.backward(loss)
            model.step()

            # Only do distributed reductions if dp_world_size > 1
            if dp_world_size > 1:
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                global_loss = loss.item() / dp_world_size
            else:
                global_loss = loss.item()

            global_distil_loss = 0
            if teacher_model is not None and dp_world_size > 1:
                dist.all_reduce(distil_loss, dist.ReduceOp.SUM)
                global_distil_loss = distil_loss.item() / dp_world_size
            elif teacher_model is not None:
                global_distil_loss = distil_loss.item()

            total_distil_loss += global_distil_loss
            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            def get_log(log_loss, log_distil_loss, log_time):
                return ("train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | "
                        "loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                ))

            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0

            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    raise NotImplementedError
                else:
                    if dp_rank == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model save to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                if dp_world_size > 1:
                    dist.barrier()

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                curr_avg_loss = evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device, adaptive_threshold)
                if "adaptive" in args.type:
                    if curr_avg_loss >= prev_avg_loss + args.loss_eps:
                        adaptive_threshold += 0.1
                        adaptive_threshold = min(adaptive_threshold, 1.0)
                        prev_avg_loss = curr_avg_loss

                model.train()

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

            if global_step > args.total_iters:
                break

    return model



def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):

    collate_fn = dataset.collate

    # Check if distributed is initialized
    dist_initialized = dist.is_initialized()
    if dist_initialized:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
    else:
        dp_world_size = 1
        dp_rank = 0
    dp_group = None
    loss_func = nn.CrossEntropyLoss()

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False
    )
    

    if dp_world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    else:
        sampler = SequentialSampler(dataset)  # If single GPU, use a simple random sampler

    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=1, num_workers=0, collate_fn=collate_fn
    )
    
    # Debug: print one batch structure
    for batch in dataloader:
        print("Batch structure:", batch)
        break

    model.eval()
    all_loss = 0.0
    step = 0
    
    all_response_ids = []
    print("tokenizer.padding_size within evluation 1:", tokenizer.padding_side)
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dp_rank != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(**model_batch).logits
            if args.model_parallel:
                raise NotImplementedError

            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))

            # Only perform generation if eval_gen is True and gen_data is not None
            if args.eval_gen and gen_data is not None:
                max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
                
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens
                )
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (args.max_length - full_ids.shape[1], 0),
                    value=tokenizer.pad_token_id,
                )
                
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)

            # Only do distributed reduction if we have more than one process
            if dp_world_size > 1:
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss = loss / dp_world_size
            
            all_loss += loss.item()
            step += 1
    
    # Only process the collected responses if generation occurred
    if args.eval_gen and len(all_response_ids) > 0:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        
        # Only perform all_gather if dp_world_size > 1
        if dp_world_size > 1:
            all_response_ids = all_gather(all_response_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
            all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
        
        responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
    else:
        responses = []

    if get_rank() == 0:
        if args.eval_gen and len(responses) > 0:
            references = dataset.answers
            responses = responses[:len(references)]
            
            res = compute_metrics(responses, references)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in responses:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        if "adaptive" in args.type:
            log_str = f"{split} | avg_loss: {avg_loss} | {res} | threshold: {adaptive_threshold}"
        else:
            log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        
    return all_loss / step




def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    dist_initialized = dist.is_initialized()
    if dist_initialized:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
    else:
        dp_world_size = 1
        dp_rank = 0  # single-process, single-gpu scenario

    # Only print args if we are rank 0 (or single process)
    if dp_rank == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    args.fp32 = not ds_config["fp16"]["enabled"]    
    args.deepspeed_config = None
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    # dp_world_size is already set above based on dist.is_initialized()
    if args.do_train:
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch
        
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    if args.teacher_model_path is not None:
        teacher_model = get_teacher_model(args, device)
    else:
        teacher_model = None
    
    if args.do_train:
        model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
    
    if args.do_eval:
        # For evaluate calls, ensure inside evaluate you also conditionally handle distributed ops
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
if __name__ == "__main__":
    main()

