"""PyTorch Dataset classes for conversational language model training."""

from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset

from hibari.data.formatting import (
    format_blended_skill_talk_example,
    format_dailydialog_example,
    format_oasst1_example,
    format_prosocial_dialog_example,
)
from hibari.data.tokenizer import HibariTokenizer


class ConversationDataset(Dataset):
    """Dataset that prepares tokenized conversational dialogues for autoregressive LM."""

    def __init__(
        self,
        conversations: Optional[List[List[Dict[str, str]]]] = None,
        tokenizer: Optional[HibariTokenizer] = None,
        max_seq_len: int = 256,
        mask_user_tokens: bool = True,
        raw_examples: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize dataset.

        Args:
            conversations: List of formatted conversations, where each conversation
                is a list of message dicts with 'role' ('system', 'user', 'assistant') and 'content'.
            tokenizer: HibariTokenizer instance.
            max_seq_len: Maximum sequence length.
            mask_user_tokens: If True, set targets to -100 for non-assistant tokens (SFT loss masking).
            raw_examples: Optional raw Blended Skill Talk examples for backward compatibility.
        """
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_user_tokens = mask_user_tokens
        self.examples: List[Dict[str, torch.Tensor]] = []

        if conversations is None:
            if raw_examples is not None:
                conversations = [format_blended_skill_talk_example(ex) for ex in raw_examples]
            else:
                conversations = []

        self._prepare(conversations)

    def _prepare(self, conversations: List[List[Dict[str, str]]]) -> None:
        asst_id = self.tokenizer.assistant_id
        eos_id = self.tokenizer.eos_id

        self.sources: List[str] = []
        for messages in conversations:
            source = "unknown"
            if isinstance(messages, dict):  # {"source": ..., "messages": [...]} records
                source = messages.get("source", "unknown")
                messages = messages["messages"]
            # Must have at least 1 user and 1 assistant message
            has_user = any(m.get("role") == "user" for m in messages)
            has_asst = any(m.get("role") == "assistant" for m in messages)
            if not (has_user and has_asst):
                continue

            text = self.tokenizer.apply_chat_template(messages)
            token_ids = self.tokenizer.encode(text)

            if len(token_ids) < 6:
                continue

            # Truncate if exceeds max_seq_len + 1 (for input and target shift)
            if len(token_ids) > self.max_seq_len + 1:
                token_ids = token_ids[: self.max_seq_len + 1]

            inp_ids = token_ids[:-1]
            tgt_ids = token_ids[1:]

            if self.mask_user_tokens:
                masked_targets = list(tgt_ids)
                in_assistant = False
                has_trainable_token = False

                for i in range(len(inp_ids)):
                    if inp_ids[i] == asst_id:
                        in_assistant = True

                    if not in_assistant:
                        masked_targets[i] = -100
                    else:
                        has_trainable_token = True

                    if in_assistant and tgt_ids[i] == eos_id:
                        in_assistant = False

                # Skip if no trainable tokens remained
                if not has_trainable_token:
                    continue

                targets_tensor = torch.tensor(masked_targets, dtype=torch.long)
            else:
                targets_tensor = torch.tensor(tgt_ids, dtype=torch.long)

            input_tensor = torch.tensor(inp_ids, dtype=torch.long)

            self.sources.append(source)
            self.examples.append(
                {
                    "input_ids": input_tensor,
                    "targets": targets_tensor,
                    "length": len(input_tensor),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
