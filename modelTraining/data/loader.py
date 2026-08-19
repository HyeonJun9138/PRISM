"""Torch Dataset / DataLoader glue around data.dataset.

The Dataset object holds only picklable state (file list + numpy index arrays
+ a plain config dict) so Windows spawn-based DataLoader workers can be used.
Episode NPZ caching happens per worker process via the lru_cache inside
data.dataset._load_episode.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data.dataset import SampleIndex, build_sample


class BlockShuffleSampler(Sampler):
    """Shuffle that respects the episode-file cache.

    A fully uniform shuffle over ~1.5M samples spread across thousands of NPZ
    files defeats any per-worker episode cache: nearly every sample decompresses
    a file, and training becomes disk-bound (measured 4x slowdown on the 5x
    dataset). This sampler shuffles at two levels instead:

      1. episode files are shuffled and packed into blocks of ``block_files``;
      2. sample order is fully shuffled WITHIN each block.

    Consecutive batches therefore touch at most ``block_files`` distinct files,
    which fits the lru_cache in data.dataset, while every epoch still sees all
    samples in a different order. Block-level shuffling is the standard
    trade-off (cf. webdataset); with blocks of 64+ files the gradient noise is
    indistinguishable from a uniform shuffle for this data.

    Deterministic: permutations derive from (seed, epoch counter).
    """

    def __init__(self, file_idx: np.ndarray, block_files: int = 64,
                 seed: int = 0):
        self.block_files = int(block_files)
        self.seed = int(seed)
        self.epoch = 0
        order = np.argsort(file_idx, kind="stable")
        sorted_files = file_idx[order]
        boundaries = np.flatnonzero(np.diff(sorted_files)) + 1
        self._per_file = np.split(order, boundaries)   # sample idx per file
        self._n = int(file_idx.shape[0])

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        self.epoch += 1
        file_order = rng.permutation(len(self._per_file))
        for start in range(0, len(file_order), self.block_files):
            block = file_order[start:start + self.block_files]
            samples = np.concatenate([self._per_file[f] for f in block])
            rng.shuffle(samples)
            yield from samples.tolist()


class TrajectoryDataset(Dataset):
    def __init__(self, index: SampleIndex, data_cfg: dict):
        self.files = index.files
        self.file_idx = index.file_idx
        self.t0 = index.t0
        self.agent = index.agent
        self.data_cfg = dict(data_cfg)

    def __len__(self) -> int:
        return int(self.file_idx.shape[0])

    def __getitem__(self, i: int) -> dict:
        s = build_sample(self.files[int(self.file_idx[i])], int(self.t0[i]),
                         int(self.agent[i]), self.data_cfg)
        return {k: torch.as_tensor(v) for k, v in s.items()}


def make_loader(index: SampleIndex, data_cfg: dict, batch_size: int,
                shuffle: bool, num_workers: int, seed: int = 0,
                max_samples: int = 0) -> DataLoader:
    """Build a DataLoader.

    shuffle=True uses BlockShuffleSampler (cache-friendly two-level shuffle;
    see its docstring). max_samples>0 instead subsamples the index each epoch
    via a RandomSampler (keeps epochs shorter on huge runs).
    """
    ds = TrajectoryDataset(index, data_cfg)
    sampler = None
    if shuffle and max_samples and max_samples < len(ds):
        g = torch.Generator(); g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(
            ds, replacement=False, num_samples=int(max_samples), generator=g)
        shuffle = False
    elif shuffle:
        sampler = BlockShuffleSampler(
            index.file_idx, block_files=int(data_cfg.get("block_files", 64)),
            seed=seed)
        shuffle = False
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                  num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  drop_last=False)
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kwargs)


def loader_with_fallback(index: SampleIndex, data_cfg: dict, batch_size: int,
                         shuffle: bool, num_workers: int, log_fn,
                         seed: int = 0, max_samples: int = 0) -> DataLoader:
    """Windows worker spawn can fail in exotic setups; fall back to in-process
    loading rather than dying, and say so in the log."""
    try:
        dl = make_loader(index, data_cfg, batch_size, shuffle, num_workers,
                         seed, max_samples)
        if num_workers > 0:     # force worker spawn now, not mid-epoch
            it = iter(dl)
            next(it)
            del it
        return dl
    except Exception as e:  # noqa: BLE001
        if num_workers == 0:
            raise
        log_fn(f"[loader] num_workers={num_workers} failed ({e!r}); "
               f"falling back to num_workers=0")
        return make_loader(index, data_cfg, batch_size, shuffle, 0,
                           seed, max_samples)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}
