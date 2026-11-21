"""Batch collator for dynamic padding and mask handling in Hibari."""

from typing import Dict, List
import torch
import torch.nn.functional as F


class DataCollatorForCausalLM:
    """Collates and pads variable length sequence examples for causal LM."""

    def __init__(self, pad_id: int = 0, ignore_index: int = -100):
        self.pad_id = pad_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        targets_list = [item["targets"] for item in batch]

        max_len = max(x.size(0) for x in input_ids_list)

        padded_inputs = []
        padded_targets = []
        attention_masks = []

        for inp, tgt in zip(input_ids_list, targets_list):
            pad_len = max_len - inp.size(0)
            if pad_len > 0:
                p_inp = F.pad(inp, (0, pad_len), value=self.pad_id)
                p_tgt = F.pad(tgt, (0, pad_len), value=self.ignore_index)
                mask = torch.cat([torch.ones(inp.size(0)), torch.zeros(pad_len)])
            else:
                p_inp = inp
                p_tgt = tgt
                mask = torch.ones(inp.size(0))

            padded_inputs.append(p_inp)
            padded_targets.append(p_tgt)
            attention_masks.append(mask)

        return {
            "input_ids": torch.stack(padded_inputs, dim=0),
            "targets": torch.stack(padded_targets, dim=0),
            "attention_mask": torch.stack(attention_masks, dim=0),
        }


class LengthGroupedBatchSampler:
    """Yield batches of similar-length examples to minimise padding.

    Indices are shuffled, split into mega-batches of ``batch_size * group``,
    sorted by length inside each mega-batch, cut into batches, and the batch
    order is shuffled again. Deterministic for a given seed and epoch.
    """

    def __init__(self, lengths, batch_size: int, group: int = 50, seed: int = 42, drop_last: bool = True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.group = group
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        idx = torch.randperm(len(self.lengths), generator=g).tolist()
        mega = self.batch_size * self.group
        batches = []
        for i in range(0, len(idx), mega):
            chunk = sorted(idx[i : i + mega], key=lambda j: self.lengths[j])
            for k in range(0, len(chunk), self.batch_size):
                b = chunk[k : k + self.batch_size]
                if len(b) == self.batch_size or not self.drop_last:
                    batches.append(b)
        order = torch.randperm(len(batches), generator=g).tolist()
        for o in order:
            yield batches[o]

    def __len__(self) -> int:
        n = len(self.lengths) // self.batch_size
        return n if self.drop_last else -(-len(self.lengths) // self.batch_size)
