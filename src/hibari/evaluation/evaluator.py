"""Model-level evaluation: token-weighted loss and perplexity, overall and per source.

``evaluate_records`` is the reusable form of what the Koto runners used to
compute themselves: conversation records (``{"source": ..., "messages": [...]}``)
are tokenised with the same SFT masking as training
(``hibari.data.ConversationDataset``: only assistant tokens count), batched
with the training collator, and scored with ``hibari.training.evaluate_loss``
(sum of token losses / number of supervised tokens). Perplexities are
therefore comparable with the training log *when the data split is the same*.
"""

from __future__ import annotations

import collections
from typing import Dict, Iterable, List, Mapping

import torch
from torch.utils.data import DataLoader

from hibari.data.collator import DataCollatorForCausalLM
from hibari.data.dataset import ConversationDataset
from hibari.data.tokenizer import HibariTokenizer
from hibari.model.llm import HibariKotoLM
from hibari.training.trainer import evaluate_loss


def evaluate_records(
    model: HibariKotoLM,
    tokenizer: HibariTokenizer,
    records: Iterable[Mapping],
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    by_source: bool = True,
) -> Dict[str, Dict[str, float]]:
    """``{"all": {...}, "<source>": {...}}`` with loss, perplexity, tokens and examples.

    Records without a ``source`` are grouped under ``"unknown"``.
    """
    recs: List[Mapping] = list(records)
    groups: Dict[str, List[Mapping]] = collections.defaultdict(list)
    for r in recs:
        groups[r.get("source", "unknown") if isinstance(r, Mapping) else "unknown"].append(r)
    out: Dict[str, Dict[str, float]] = {}
    collate = DataCollatorForCausalLM(tokenizer.pad_id)
    device = torch.device(device)

    def score(items: List[Mapping]) -> Dict[str, float] | None:
        ds = ConversationDataset(items, tokenizer, max_seq_len=model.config.max_seq_len)
        if len(ds) == 0:
            return None
        m = evaluate_loss(model, DataLoader(ds, batch_size=batch_size, collate_fn=collate), device)
        return {"loss": round(m["eval_loss"], 4), "perplexity": round(m["eval_perplexity"], 2),
                "tokens": m["eval_tokens"], "examples": len(ds)}

    total = score(recs)
    if total:
        out["all"] = total
    if by_source:
        for src in sorted(groups):
            s = score(groups[src])
            if s:
                out[src] = s
    return out
