"""Tokenizer implementation for Hibari models."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]


class HibariTokenizer:
    """Byte-level BPE Tokenizer tailored for Hibari conversational LLMs."""

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is None:
            self._tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
            self._tokenizer.normalizer = normalizers.NFKC()
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self._tokenizer.decoder = decoders.ByteLevel()
        else:
            self._tokenizer = tokenizer

        self._special_token_to_id: Dict[str, int] = {}
        self._refresh_special_tokens()

    def _refresh_special_tokens(self) -> None:
        self._special_token_to_id = {
            tok: self._tokenizer.token_to_id(tok)
            for tok in SPECIAL_TOKENS
            if self._tokenizer.token_to_id(tok) is not None
        }

    @property
    def pad_id(self) -> int:
        return self._special_token_to_id.get("<|pad|>", 0)

    @property
    def unk_id(self) -> int:
        return self._special_token_to_id.get("<|unk|>", 1)

    @property
    def bos_id(self) -> int:
        return self._special_token_to_id.get("<|bos|>", 2)

    @property
    def eos_id(self) -> int:
        return self._special_token_to_id.get("<|eos|>", 3)

    @property
    def system_id(self) -> int:
        return self._special_token_to_id.get("<|system|>", 4)

    @property
    def user_id(self) -> int:
        return self._special_token_to_id.get("<|user|>", 5)

    @property
    def assistant_id(self) -> int:
        return self._special_token_to_id.get("<|assistant|>", 6)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        return self._tokenizer.get_vocab()

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int = 1024,
        min_frequency: int = 2,
    ) -> None:
        """Train BPE tokenizer on text corpus."""
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._refresh_special_tokens()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = self._tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Format a conversation turn into Hibari chat template.

        Format:
        <|system|>
        {content}<|eos|>
        <|user|>
        {content}<|eos|>
        <|assistant|>
        {content}<|eos|>
        """
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "system":
                formatted += f"<|system|>\n{content}<|eos|>\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}<|eos|>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}<|eos|>\n"

        if add_generation_prompt:
            formatted += "<|assistant|>\n"

        return formatted

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path / "tokenizer.json"))
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "special_tokens": SPECIAL_TOKENS,
                    "pad_token": "<|pad|>",
                    "eos_token": "<|eos|>",
                    "bos_token": "<|bos|>",
                    "unk_token": "<|unk|>",
                    "vocab_size": self.vocab_size,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path) -> "HibariTokenizer":
        path = Path(directory)
        tokenizer_file = path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_file}")
        t = Tokenizer.from_file(str(tokenizer_file))
        return cls(t)
