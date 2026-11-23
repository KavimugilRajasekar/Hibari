"""High-level inference engine for Hibari."""

from pathlib import Path
from typing import Optional
import torch

from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.conversation import HibariChatSession, TaskResult
from hibari.inference.generation import generate_text
from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM
from hibari.training.checkpoint import load_checkpoint


class InferenceEngine:
    """Wraps model and tokenizer for convenient text generation and chat."""

    def __init__(
        self,
        model: HibariKotoLM,
        tokenizer: HibariTokenizer,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.profile = None  # set by from_profile

    @classmethod
    def from_profile(cls, profile, device: str | torch.device = "cpu") -> "InferenceEngine":
        """Load the checkpoint and tokenizer named by a release profile
        (``hibari.release.ModelProfile`` or a path to its YAML). Sessions created
        afterwards use the profile's system prompt and inference settings."""
        from hibari.release.profile import ModelProfile

        if not isinstance(profile, ModelProfile):
            profile = ModelProfile.load(profile)
        engine = cls.from_checkpoint(profile.resolve(profile.checkpoint), profile.resolve(profile.tokenizer_dir), device)
        if engine.tokenizer.vocab_size != engine.model.config.vocab_size:
            raise ValueError("Tokenizer and checkpoint vocabulary sizes differ")
        engine.profile = profile
        return engine

    def reply_settings(self) -> dict:
        """Decoding keyword arguments for ``HibariChatSession.reply`` (profile defaults or {})."""
        return self.profile.inference.reply_kwargs() if self.profile else {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "InferenceEngine":
        loaded = load_checkpoint(checkpoint_path, device=device)
        tokenizer = HibariTokenizer.load(tokenizer_path)
        return cls(model=loaded["model"], tokenizer=tokenizer, device=device)

    def create_chat_session(
        self, system_prompt: Optional[str] = None, persona: Optional[str] = None, **overrides
    ) -> HibariChatSession:
        """New session. With a profile, its system prompt, identity terms and
        application-layer settings are used unless overridden."""
        kwargs = {}
        if self.profile is not None:
            kwargs = dict(self.profile.inference.session_kwargs(), identity_terms=self.profile.identity.terms)
            system_prompt = system_prompt or self.profile.system_prompt
        kwargs.update(overrides)
        return HibariChatSession(
            model=self.model,
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            persona=persona,
            device=self.device,
            **kwargs,
        )

    def rewrite(self, task: str, text: str, system_prompt: Optional[str] = None) -> TaskResult:
        """One-shot rewriting task (rephrase, correct_grammar, formal, casual,
        shorten, simplify, polite); see ``hibari.inference.tasks``."""
        session = self.create_chat_session(system_prompt=system_prompt)
        return session.run_task(task, text, add_to_history=False)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        seed: Optional[int] = None,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        return generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=self.device,
            seed=seed,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
