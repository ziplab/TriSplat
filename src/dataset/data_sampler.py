# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Random sampling under a constraint
# --------------------------------------------------------
import torch
from torch.utils.data import Sampler, BatchSampler
from torch import Tensor


def custom_collate_fn(batch):
    elem = batch[0]

    if isinstance(elem, Tensor):
        # Some environments surface tensors backed by non-resizable storages
        # (for example tensors originating from numpy/PIL conversions). Clone
        # before stacking so worker-side shared-memory collation stays stable.
        return torch.stack([item.contiguous().clone() for item in batch], dim=0)

    if isinstance(elem, dict):
        return {
            key: custom_collate_fn([sample[key] for sample in batch])
            for key in elem
        }

    if isinstance(elem, tuple):
        return tuple(custom_collate_fn(list(items)) for items in zip(*batch))

    if isinstance(elem, list):
        return [custom_collate_fn(list(items)) for items in zip(*batch)]

    return torch.utils.data.default_collate(batch)

class BatchedRandomSampler(Sampler):
    def __init__(
            self, dataset, max_img_per_gpu, num_context_views, world_size=1, rank=0,
            drop_last=True
    ):
        self.dataset = dataset
        self.max_img_per_gpu = max_img_per_gpu
        self.num_context_views = num_context_views
        self.world_size = world_size
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        # Deterministic shuffling
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        
        # Subsample for DDP
        if self.world_size > 1:
            num_samples = len(self.dataset) // self.world_size
            if self.drop_last:
                indices = indices[:num_samples * self.world_size]
            indices = indices[self.rank:len(indices):self.world_size]
        
        current_idx = 0
        while current_idx < len(indices):
            if isinstance(self.num_context_views, (list, tuple)):
                num_imgs = torch.randint(
                    self.num_context_views[0], 
                    self.num_context_views[1] + 1, 
                    (), 
                    generator=g
                ).item()
            else:
                num_imgs = self.num_context_views
            
            batch_size = max(1, self.max_img_per_gpu // num_imgs)
            
            batch_indices = indices[current_idx : current_idx + batch_size]
            
            if self.drop_last and len(batch_indices) < batch_size:
                break
            
            if len(batch_indices) == 0:
                break

            yield [(idx, num_imgs) for idx in batch_indices]
            
            current_idx += batch_size

    def __len__(self):
        # Approximate length
        if isinstance(self.num_context_views, (list, tuple)):
            avg_imgs = (self.num_context_views[0] + self.num_context_views[1]) / 2
        else:
            avg_imgs = self.num_context_views
        avg_batch_size = max(1, self.max_img_per_gpu // avg_imgs)
        num_samples = len(self.dataset)
        if self.world_size > 1:
            num_samples //= self.world_size
        return int(num_samples // avg_batch_size)

class MixedBatchSampler(BatchSampler):
    def __init__(
            self, src_dataset_ls, max_img_per_gpu, num_context_views, world_size=1, rank=0, prob=None,
            generator=None, sampler=None, batch_size=None, drop_last=None, step_tracker=None
    ):
        self.src_dataset_ls = src_dataset_ls
        self.max_img_per_gpu = max_img_per_gpu
        self.num_context_views = num_context_views
        self.world_size = world_size
        self.rank = rank
        self.prob = prob
        self.generator = generator
        self.step_tracker = step_tracker
        
        self.dataset_lengths = [len(ds) for ds in src_dataset_ls]
        self.cum_dataset_length = [0]
        for l in self.dataset_lengths:
            self.cum_dataset_length.append(self.cum_dataset_length[-1] + l)
        
        self.samplers = []
        for i, ds in enumerate(src_dataset_ls):
            if isinstance(num_context_views, list) and len(num_context_views) == len(src_dataset_ls):
                 ncv = num_context_views[i]
            else:
                 ncv = num_context_views

            sampler = BatchedRandomSampler(
                ds, max_img_per_gpu, ncv, world_size, rank
            )
            self.samplers.append(sampler)
            
        self.epoch = 0
        self.batches = []
        self._generate_batches(self.epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _generate_batches(self, epoch):
        for sampler in self.samplers:
            sampler.set_epoch(epoch)
        
        batches = []
        for i, sampler in enumerate(self.samplers):
            shift = self.cum_dataset_length[i]
            for batch in sampler:
                # batch is list of (idx, num_imgs)
                shifted_batch = [(idx + shift, num_imgs) for idx, num_imgs in batch]
                batches.append(shifted_batch)
            
        # Shuffle batches
        g = torch.Generator()
        g.manual_seed(epoch)
        indices = torch.randperm(len(batches), generator=g).tolist()
        self.batches = [batches[i] for i in indices]

    def __iter__(self):
        if self.step_tracker is not None:
            step = self.step_tracker.get_step()
            # Calculate epoch based on step and dataset length
            # Assuming len(self) is the number of batches per epoch
            epoch = step // len(self)
            if epoch != self.epoch:
                self.epoch = epoch
                self._generate_batches(self.epoch)
        else:
            # Fallback or rely on set_epoch
            pass
            
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
