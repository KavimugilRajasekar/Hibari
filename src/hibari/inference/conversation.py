"""Multi-turn conversation session manager for Hibari Koto.

Pipeline for each user turn:

1. Deterministic safety check (``hibari.inference.safety.assess_message``).
   Self-harm, violent intent and harmful-instruction requests receive a fixed,
   reviewed response; the model is not called. This layer is application-side
   only and does not exist inside the GGUF.
2. Otherwise the conversation is formatted with the chat template, the oldest
   user/assistant turns are dropped until the prompt fits the context window
   (the system prompt is always kept), and the model generates the reply.
3. The reply is cleaned (chat markers removed, emojis capped, a dangling
   unfinished sentence trimmed). No content is added.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from hibari.data.formatting import DEFAULT_SYSTEM_PROMPT
from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.generation import generate_tokens
from hibari.inference.safety import (
    assess_message,
    clean_response,
    get_safety_response,
)
from hibari.model.llm import HibariKotoLM

EMPTY_INPUT_RESPONSE = "I didn't catch that. What would you like to talk about?"


class HibariChatSession:
    """Conversational session managing dialogue history and Hibari Koto replies."""

    def __init__(
        self,
        model: HibariKotoLM,
        tokenizer: HibariTokenizer,
        system_prompt: Optional[str] = None,
        persona: Optional[str] = None,
        device: str | torch.device = "cpu",
        enable_safety: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.enable_safety = enable_safety
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        if persona:
            self.system_prompt = f"{self.system_prompt} {persona}"
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.last_safety_category: Optional[str] = None

    def add_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message.strip()})

    def build_prompt_ids(self, max_new_tokens: int) -> List[int]:
        """Chat-format the history, dropping the oldest turns to fit the context."""
        budget = self.model.config.max_seq_len - max_new_tokens
        system = self.messages[:1]
        turns = self.messages[1:]
        while True:
            text = self.tokenizer.apply_chat_template(
                system + turns, add_generation_prompt=True
            )
            ids = self.tokenizer.encode(text)
            if len(ids) <= budget or len(turns) <= 1:
                break
            # drop the oldest turn; keep the history starting with a user turn
            turns = turns[1:]
            while turns and turns[0]["role"] != "user" and len(turns) > 1:
                turns = turns[1:]
        if len(ids) > budget:
            # A single very long user message: keep the template head (system)
            # and the most recent tokens of the message.
            head = self.tokenizer.encode(
                self.tokenizer.apply_chat_template(system)
            )
            tail_budget = budget - len(head)
            if tail_budget >= 8:
                ids = head + ids[-tail_budget:]
            else:  # pathological: even the system prompt does not fit
                ids = ids[-budget:]
        return ids

    def reply(
        self,
        user_message: str,
        max_new_tokens: int = 64,
        temperature: float = 0.5,
        top_k: int = 25,
        top_p: float = 0.85,
        repetition_penalty: float = 1.15,
        seed: Optional[int] = None,
    ) -> str:
        """Send a user message and return Hibari's reply."""
        user_message = (user_message or "").strip()
        if not user_message:
            return EMPTY_INPUT_RESPONSE

        self.add_user_message(user_message)

        assessment = assess_message(user_message) if self.enable_safety else None
        self.last_safety_category = assessment.category.value if assessment else None
        if assessment is not None and assessment.is_flagged:
            response = get_safety_response(assessment.category)
            self.messages.append({"role": "assistant", "content": response})
            return response

        prompt_ids = self.build_prompt_ids(max_new_tokens)
        new_ids = generate_tokens(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt="",
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=self.device,
            seed=seed,
        )
        response = clean_response(
            self.tokenizer.decode(new_ids, skip_special_tokens=True)
        )
        if not response:
            response = "Sorry, I'm not sure how to answer that. Could you say it another way?"

        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset(self, new_persona: Optional[str] = None) -> None:
        base = DEFAULT_SYSTEM_PROMPT if new_persona is None else f"{DEFAULT_SYSTEM_PROMPT} {new_persona}"
        self.system_prompt = base
        self.messages = [{"role": "system", "content": base}]
