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

Optional (enabled by a model's release profile, see ``hibari.release``):

* ``normalize_input`` — the model sees the user message with chat shorthand
  expanded (``hibari.inference.normalization``); the safety layer and the
  rewriting router always see the original text.
* ``num_candidates > 1`` — the model generates several replies (the first
  with the caller's decoding settings, the rest sampled with fixed seeds) and
  ``hibari.inference.reranking`` picks the one that best fits the message:
  fluency, context relevance (PMI against an empty user turn) and named
  appropriateness rules. Every reply is still written by the model.
  ``last_reply_info`` records all candidates and why the winner won.

Rewriting tasks (rephrase, grammar correction, formal / casual / polite,
shorten, simplify) are recognised by ``hibari.inference.tasks`` and sent in
the canonical ``Task: ...\nInput: ...`` form that the model is trained on,
without earlier chat turns (a rewrite depends only on its input), with
deterministic task decoding. ``run_task`` is the direct API for the same path.
If the best candidate drops a protected term (number, name, time, technical
token) a few seeded samples are tried and the candidate that keeps the most
protected terms wins; the result reports any terms still missing. Text that
does not fit the context window with room for the reply is rewritten in
consecutive groups of sentences; a single sentence that is too long gets a
fixed message instead of a truncated prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from hibari.data.formatting import DEFAULT_SYSTEM_PROMPT
from hibari.data.tokenizer import HibariTokenizer, escape_special_tokens
from hibari.inference.generation import generate_batch, generate_tokens, reply_reserve
from hibari.inference.intent import Intent, MessageAnalysis, analyze_message
from hibari.inference.normalization import normalize_informal
from hibari.inference.reranking import Candidate, CriticWeights, ReplyCritic, reply_logprobs, score_candidates
from hibari.inference.safety import (
    assess_message,
    clean_response,
    get_safety_response,
)
from hibari.inference.tasks import (
    TASK_DECODING,
    TASKS,
    format_task_request,
    missing_protected_terms,
    parse_task_request,
    task_max_new_tokens,
)
from hibari.model.llm import HibariKotoLM

EMPTY_INPUT_RESPONSE = "I didn't catch that. What would you like to talk about?"

# Sampling used for the extra candidates when num_candidates > 1.
DEFAULT_CANDIDATE_SAMPLING = dict(temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1.1)
EMAIL_MAX_NEW_TOKENS = 160


TOO_LONG_TASK_RESPONSE = "That text is too long for me to rewrite in one piece. Could you send it in shorter parts?"


def empty_task_response(task: str) -> str:
    return f"Please share the text you'd like me to {TASKS[task].verb}."


@dataclass
class TaskResult:
    task: str
    input: str
    output: str
    missing_terms: List[str] = field(default_factory=list)
    candidates_tried: int = 0
    safety_category: Optional[str] = None
    chunks: int = 0


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
        route_tasks: bool = True,
        normalize_input: bool = False,
        num_candidates: int = 1,
        candidate_sampling: Optional[Dict] = None,
        identity_terms: Sequence[str] = (),
        critic_weights: Optional[CriticWeights] = None,
        intent_candidates: Optional[Dict[str, int]] = None,
    ):
        self.model = model
        self.normalize_input = normalize_input
        self.num_candidates = max(1, int(num_candidates))
        self.candidate_sampling = dict(candidate_sampling or DEFAULT_CANDIDATE_SAMPLING)
        # per-intent pool size, e.g. {"emotional_disclosure": 16}; only used when num_candidates > 1
        self.intent_candidates = dict(intent_candidates or {})
        self.critic = ReplyCritic(identity_terms=identity_terms, weights=critic_weights)
        self.route_tasks = route_tasks
        self.tokenizer = tokenizer
        self.device = device
        self.enable_safety = enable_safety
        self.base_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.system_prompt = self.base_system_prompt
        if persona:
            self.system_prompt = f"{self.system_prompt} {persona}"
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.last_safety_category: Optional[str] = None
        self.last_task: Optional[TaskResult] = None
        # True when the last generated reply ended with a stop token rather
        # than hitting the token limit (None if no generation happened).
        self.last_stopped_on_eos: Optional[bool] = None
        # Debug record of the last reply: analysis, model input, candidates.
        self.last_reply_info: Dict = {}

    def add_user_message(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message.strip()})

    def build_prompt_ids(self, max_new_tokens: int, messages: Optional[List[Dict[str, str]]] = None) -> List[int]:
        """Chat-format the history, dropping the oldest turns to fit the context."""
        messages = self.messages if messages is None else messages
        # Same reserve as generation (``reply_reserve``): at most half the window. Reserving the full
        # max_new_tokens cut the start of long requests (e-mail: 160 of 256) although generation would
        # have accepted them, and a max_new_tokens >= the window gave a zero or negative budget.
        budget = self.model.config.max_seq_len - reply_reserve(self.model.config.max_seq_len, max_new_tokens)
        system = messages[:1]
        turns = messages[1:]
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
            # A single very long user message: keep the system turn and the
            # user header, the most recent tokens of the message, and the
            # end-of-turn + assistant header, so the model still sees a
            # well-formed user turn.
            last = turns[-1] if turns else {"role": "user", "content": ""}
            head = self.tokenizer.encode(self.tokenizer.apply_chat_template(system) + "<|user|>\n")
            tail = self.tokenizer.encode("<|eos|>\n<|assistant|>\n")
            content = self.tokenizer.encode(escape_special_tokens(last.get("content", "").strip()))
            keep = budget - len(head) - len(tail)
            if keep >= 8:
                ids = head + content[-keep:] + tail
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
        no_repeat_ngram_size: int = 0,
    ) -> str:
        """Send a user message and return Hibari's reply."""
        user_message = (user_message or "").strip()
        self.last_task = None
        self.last_stopped_on_eos = None
        self.last_reply_info = {}
        if not user_message:
            return EMPTY_INPUT_RESPONSE

        request = parse_task_request(user_message) if self.route_tasks else None
        if request is not None:
            return self.run_task(request.task, request.text).output

        analysis = analyze_message(user_message)
        model_text = normalize_informal(user_message) if self.normalize_input else user_message
        self.last_reply_info = {"analysis": analysis.to_dict(), "model_input": model_text, "candidates": []}
        self.add_user_message(model_text)

        assessment = assess_message(user_message) if self.enable_safety else None
        self.last_safety_category = assessment.category.value if assessment else None
        if assessment is not None and assessment.is_flagged:
            response = get_safety_response(assessment.category)
            self.messages.append({"role": "assistant", "content": response})
            return response

        if analysis.intent is Intent.EMAIL_REQUEST:
            max_new_tokens = max(max_new_tokens, EMAIL_MAX_NEW_TOKENS)
        prompt_ids = self.build_prompt_ids(max_new_tokens)
        decoding = dict(temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size)
        if self.num_candidates > 1:
            response, eos = self._reply_with_candidates(prompt_ids, max_new_tokens, decoding, seed, analysis)
        else:
            response, eos, _ = self._generate_reply(prompt_ids, max_new_tokens, decoding, seed)
        self.last_stopped_on_eos = eos
        if not response:
            response = "Sorry, I'm not sure how to answer that. Could you say it another way?"

        self.messages.append({"role": "assistant", "content": response})
        return response

    def _generate_reply(self, prompt_ids: List[int], max_new_tokens: int, decoding: Dict, seed: Optional[int]) -> tuple:
        gen_stats: Dict = {}
        new_ids = generate_tokens(
            model=self.model, tokenizer=self.tokenizer, prompt="", prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens, device=self.device, seed=seed, stats=gen_stats, **decoding,
        )
        text = clean_response(self.tokenizer.decode(new_ids, skip_special_tokens=True))
        return text, gen_stats.get("stop_reason") == "stop_token", new_ids

    def _generic_prompt_ids(self) -> List[int]:
        """System prompt + an empty user turn: the baseline for context relevance."""
        msgs = self.messages[:1] + [{"role": "user", "content": ""}]
        return self.tokenizer.encode(self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True))

    def _reply_with_candidates(self, prompt_ids: List[int], max_new_tokens: int, decoding: Dict,
                               seed: Optional[int], analysis: MessageAnalysis) -> tuple:
        base_seed = 0 if seed is None else seed
        cands: List[Candidate] = []
        seen = set()
        pool = max(self.num_candidates, int(self.intent_candidates.get(analysis.intent.value, 0)))
        self.last_reply_info["pool_size"] = pool
        decs = [decoding] + [dict(self.candidate_sampling, no_repeat_ngram_size=decoding.get("no_repeat_ngram_size", 0))] * (pool - 1)
        seeds = [seed] + [base_seed + 1000 + i for i in range(1, pool)]
        # one batched pass: the prompt is encoded once and all candidates decode together
        results = generate_batch(self.model, self.tokenizer, prompt_ids, decs, seeds, max_new_tokens, self.device)
        for i, (ids, reason) in enumerate(results):
            text = clean_response(self.tokenizer.decode(ids, skip_special_tokens=True))
            eos = reason == "stop_token"
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            scored = self.tokenizer.encode(text) + ([self.tokenizer.eos_id] if eos else [])
            cands.append(Candidate(text=text, token_ids=scored, stopped_on_eos=eos,
                                   source="primary" if i == 0 else f"sample:{seeds[i]}", n_scored_tokens=len(scored)))
        ctx_lp = reply_logprobs(self.model, prompt_ids, [c.token_ids for c in cands], self.device)
        gen_lp = reply_logprobs(self.model, self._generic_prompt_ids(), [c.token_ids for c in cands], self.device)
        context = " ".join(m["content"] for m in self.messages)
        previous = [m["content"] for m in self.messages if m["role"] == "assistant"]
        # a repeated question deserves the same answer: only penalise repeats otherwise
        users = [re.sub(r"\W+", " ", m["content"].lower()).strip() for m in self.messages if m["role"] == "user"]
        if users and users[-1] in users[:-1]:
            previous = []
        for c, a, b in zip(cands, ctx_lp, gen_lp):
            c.logprob, c.generic_logprob = a, b
            c.adjustments = self.critic.judge(c.text, analysis, context, c.stopped_on_eos, previous_replies=previous)
        ranked = score_candidates(cands, self.critic.w)
        self.last_reply_info["candidates"] = [c.to_dict() for c in ranked]
        best = ranked[0]
        return best.text, best.stopped_on_eos

    def _task_candidate(self, prompt_ids: List[int], max_new: int, decoding: Dict, seed: Optional[int]) -> tuple:
        gen_stats: Dict = {}
        new_ids = generate_tokens(
            model=self.model, tokenizer=self.tokenizer, prompt="", prompt_ids=prompt_ids,
            max_new_tokens=max_new, device=self.device, seed=seed, stats=gen_stats, **decoding,
        )
        text = clean_response(self.tokenizer.decode(new_ids, skip_special_tokens=True), max_emojis=0)
        return text, gen_stats.get("stop_reason") == "stop_token"

    def _task_prompt(self, task: str, text: str) -> tuple:
        """(prompt_ids, max_new_tokens) for one rewrite, or None if the request
        plus a minimal reply does not fit the context window."""
        request = format_task_request(task, text)
        msgs = self.messages[:1] + [{"role": "user", "content": request}]
        ids = self.tokenizer.encode(self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True))
        n_in = len(self.tokenizer.encode(text))
        need = max(16, int(n_in * (0.6 if task in ("shorten", "simplify") else 1.0)) + 8)
        avail = self.model.config.max_seq_len - len(ids)
        if avail < need:
            return None
        return ids, min(task_max_new_tokens(n_in, task), avail)

    def _task_chunks(self, task: str, text: str) -> Optional[List[str]]:
        """The whole text if it fits; otherwise consecutive groups of sentences
        that each fit. None if a single sentence is already too long."""
        if self._task_prompt(task, text) is not None:
            return [text]
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        chunks: List[str] = []
        current = ""
        for s in sentences:
            candidate = f"{current} {s}".strip()
            if self._task_prompt(task, candidate) is not None:
                current = candidate
                continue
            if not current or self._task_prompt(task, s) is None:
                return None
            chunks.append(current)
            current = s
        if current:
            chunks.append(current)
        return chunks

    def _rewrite_one(self, task: str, text: str, retries: int, retry_temperature: float) -> tuple:
        """(output, candidates_tried, stopped_on_eos) for a text that fits."""
        prompt_ids, max_new = self._task_prompt(task, text)
        best, best_eos = self._task_candidate(prompt_ids, max_new, TASK_DECODING, None)
        best_missing = missing_protected_terms(text, best)
        tried = 1
        for i in range(retries if best_missing else 0):
            dec = dict(TASK_DECODING, temperature=retry_temperature, top_k=20, top_p=0.9)
            cand, cand_eos = self._task_candidate(prompt_ids, max_new, dec, seed=i)
            tried += 1
            miss = missing_protected_terms(text, cand)
            if cand and len(miss) < len(best_missing):
                best, best_missing, best_eos = cand, miss, cand_eos
            if not best_missing:
                break
        return best, tried, best_eos

    def run_task(
        self,
        task: str,
        text: str,
        retries: int = 3,
        retry_temperature: float = 0.3,
        add_to_history: bool = True,
    ) -> TaskResult:
        """Rewrite ``text`` with ``task`` (see ``hibari.inference.tasks.TASKS``).

        The request and the result are appended to the conversation history so
        later turns can refer to them; the prompt itself contains only the
        system prompt and the task request.
        """
        if task not in TASKS:
            raise KeyError(f"Unknown task {task!r}; expected one of {sorted(TASKS)}")
        text = (text or "").strip()
        request = format_task_request(task, text)
        result = TaskResult(task=task, input=text, output="")
        self.last_stopped_on_eos = None

        if not text:
            result.output = empty_task_response(task)
        else:
            assessment = assess_message(text) if self.enable_safety else None
            result.safety_category = assessment.category.value if assessment else None
            if assessment is not None and assessment.is_flagged:
                result.output = get_safety_response(assessment.category)
            else:
                chunks = self._task_chunks(task, text)
                if chunks is None:
                    result.output = TOO_LONG_TASK_RESPONSE
                else:
                    outs, tried, eos = [], 0, True
                    for chunk in chunks:
                        o, n, e = self._rewrite_one(task, chunk, retries, retry_temperature)
                        outs.append(o)
                        tried += n
                        eos = eos and e
                    joined = " ".join(o for o in outs if o)
                    result.output = joined or "Sorry, I couldn't rewrite that. Could you try again?"
                    result.missing_terms = missing_protected_terms(text, joined)
                    result.candidates_tried = tried
                    result.chunks = len(chunks)
                    self.last_stopped_on_eos = eos

        if self.last_stopped_on_eos is None:
            self.last_stopped_on_eos = True  # fixed message, no generation
        self.last_safety_category = result.safety_category
        self.last_task = result
        if add_to_history:
            self.messages.append({"role": "user", "content": request})
            self.messages.append({"role": "assistant", "content": result.output})
        return result

    def reset(self, new_persona: Optional[str] = None) -> None:
        """Clear the history. The system prompt given to the constructor is kept;
        ``new_persona`` (if any) is appended to it."""
        base = self.base_system_prompt if new_persona is None else f"{self.base_system_prompt} {new_persona}"
        self.system_prompt = base
        self.messages = [{"role": "system", "content": base}]
        self.last_safety_category = None
        self.last_task = None
        self.last_stopped_on_eos = None
