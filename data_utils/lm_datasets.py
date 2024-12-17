import random
import torch
import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size, barrier
from utils import print_rank
from utils import save_rank


class LMTrainDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.ratio = ratio
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", get_rank(), get_world_size())

        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        
        print_rank(len(self.lm_ctx))
        if num == -1:
            self.num = len(self.lm_ctx)
        else:
            self.num = num

        print_rank(f"Num LM instances: {len(self.lm_ctx)}")

    def __len__(self):
        return len(self.lm_ctx)
   
    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of length {len(self)}.")
        return self._get_lm(index)
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }

    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        prompt = None
        if 65535 in input_ids:
            source_len = np.where(input_ids==65535)[0][0]
            prompt = input_ids[:source_len]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0
        
        if prompt is not None:
            gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        if gen_data is not None:
            for k, v in gen_data.items():
                gen_data[k] = v.to(device)

        return model_data, no_model_data, gen_data

    def collate(self, samples):
        # Filter out samples that have no tokens
        samples = [s for s in samples if s is not None and len(s.get("input_ids", [])) > 0]
    
        # If no valid samples remain, raise an error or return None
        if not samples:
            raise ValueError("All samples in this batch are empty after tokenization.")
    
        # Dynamically calculate max_length from input_ids
        max_length = max(len(s["input_ids"]) for s in samples)
    
        bs = len(samples)
        # Initialize model_data
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
            "position_ids": torch.arange(max_length).unsqueeze(0).expand(bs, -1),
        }
    
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length),
        }
    
        # Check for gen_input_ids
        max_prompt_length = max((len(s.get("gen_input_ids", [])) for s in samples), default=0)
    
        # If we have gen_input_ids, create gen_data
        if max_prompt_length > 0:
            gen_data = {
                "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
                "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
            }
        else:
            # If no gen_input_ids are present, set gen_data to None
            gen_data = None
    
        # Fill model_data and no_model_data
        for i, samp in enumerate(samples):
            seq_len = len(samp["input_ids"])
            model_data["input_ids"][i, :seq_len] = torch.tensor(samp["input_ids"], dtype=torch.long)
            model_data["attention_mask"][i, :seq_len] = 1.0
    
            # If we have gen_data, we need to process it as well
            # (Only if gen_data is not None and samp has gen_input_ids)
            if gen_data is not None and "gen_input_ids" in samp and len(samp["gen_input_ids"]) > 0:
                gen_len = len(samp["gen_input_ids"])
                gen_data["input_ids"][i, :gen_len] = torch.tensor(samp["gen_input_ids"], dtype=torch.long)
                gen_data["attention_mask"][i, :gen_len] = 1.0
    
            # Process LM-related fields
            try:
                self._process_lm(i, samp, model_data, no_model_data, gen_data)
            except Exception as e:
                raise RuntimeError(f"Error in _process_lm for sample {i}: {e}")
        return model_data, no_model_data,gen_data

    
    
