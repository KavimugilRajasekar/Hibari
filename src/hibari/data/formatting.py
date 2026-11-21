"""Conversation formatting utilities for Multi-Dataset Hibari dialogues.

Supports:
1. anezatra/blended-skill-talk (Blended Skill Talk)
2. roskoN/dailydialog (DailyDialog)
3. allenai/prosocial-dialog (ProsocialDialog)
4. OpenAssistant/oasst1 (OpenAssistant)
5. Estwld/empathetic_dialogues_llm (EmpatheticDialogues)
6. HuggingFaceTB/everyday-conversations-llama3.1-2k
7. HuggingFaceTB/smol-smoltalk (short single-turn subsets)
8. hibari.data.curated (hand-written identity / personality / safety set)

Every conversation is normalised by ``normalize_conversation`` (clean text,
strict user/assistant alternation, ends on an assistant turn) and split into
context-sized windows by ``window_conversation`` so that no training example
is cut in the middle of a reply.
"""

import re
from typing import Any, Callable, Dict, List, Optional


# Used for every training example and at inference (Python, llama.cpp, Ollama).
DEFAULT_SYSTEM_PROMPT = (
    "You are Hibari Koto, a friendly and calm AI companion. "
    "Keep replies short, kind and honest."
)

_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F000-\U0001F2FF]\ufe0f?")


def clean_text(text: str, strip_emojis: bool = False) -> str:
    """Fix tokenised-corpus spacing (" ." / " ' s") and normalise quotes/whitespace."""
    if not isinstance(text, str):
        return ""
    t = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\r\n", "\n")
    if strip_emojis:
        t = _EMOJI_RE.sub("", t)
    t = re.sub(r" +([.,!?;:%)])", r"\1", t)          # "hello ." -> "hello."
    t = re.sub(r"([(]) +", r"\1", t)
    t = re.sub(r" ' (s|t|m|re|ve|ll|d)\b", r"'\1", t)  # "it ' s" -> "it's"
    t = re.sub(r"(\w) '(s|t|m|re|ve|ll|d)\b", r"\1'\2", t)
    t = re.sub(r"\b(\w+) n't\b", r"\1n't", t)        # "do n't" -> "don't"
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def normalize_conversation(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    strip_emojis: bool = True,
) -> List[Dict[str, str]]:
    """Return [system, user, assistant, user, assistant, ...] or [] if unusable.

    * replaces any source system prompt with Hibari's,
    * cleans text, drops empty turns, merges consecutive same-role turns,
    * drops leading assistant turns and trailing user turns.
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    turns: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = clean_text(m.get("content", ""), strip_emojis=strip_emojis and role == "assistant")
        if not content:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + content
        else:
            turns.append({"role": role, "content": content})
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    while turns and turns[-1]["role"] != "assistant":
        turns.pop()
    if len(turns) < 2:
        return []
    return [{"role": "system", "content": sys_prompt}] + turns


def window_conversation(
    messages: List[Dict[str, str]],
    count_tokens: Callable[[List[Dict[str, str]]], int],
    max_tokens: int,
) -> List[List[Dict[str, str]]]:
    """Split a normalised conversation into windows that fit ``max_tokens``.

    Each window is [system, user, assistant, ...] and ends on an assistant
    turn. User/assistant pairs that do not fit even alone are dropped.
    """
    if not messages:
        return []
    system, turns = messages[0], messages[1:]
    pairs = [turns[i : i + 2] for i in range(0, len(turns) - 1, 2)]
    windows: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    for pair in pairs:
        candidate = [system] + current + pair
        if count_tokens(candidate) <= max_tokens:
            current = current + pair
            continue
        if current:
            windows.append([system] + current)
        current = pair if count_tokens([system] + pair) <= max_tokens else []
    if current:
        windows.append([system] + current)
    return windows


def format_blended_skill_talk_example(
    example: Dict[str, Any],
    include_personas: bool = False,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Convert a row from anezatra/blended-skill-talk into chat messages."""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    personas = example.get("personas") or []
    if include_personas and personas:
        persona_str = " ".join([p.strip() for p in personas if p and isinstance(p, str)])
        if persona_str:
            sys_prompt = f"{sys_prompt} Persona: {persona_str}"

    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]

    free_msgs = example.get("free_messages") or []
    guided_msgs = example.get("guided_messages") or []

    prev_utts = example.get("previous_utterance") or []
    if prev_utts and len(prev_utts) >= 2:
        for i in range(0, len(prev_utts) - 1, 2):
            u = prev_utts[i].strip() if prev_utts[i] else ""
            a = prev_utts[i + 1].strip() if prev_utts[i + 1] else ""
            if u:
                messages.append({"role": "user", "content": u})
            if a:
                messages.append({"role": "assistant", "content": a})

    for user_msg, asst_msg in zip(free_msgs, guided_msgs):
        u_text = user_msg.strip() if isinstance(user_msg, str) else ""
        a_text = asst_msg.strip() if isinstance(asst_msg, str) else ""
        if u_text:
            messages.append({"role": "user", "content": u_text})
        if a_text:
            messages.append({"role": "assistant", "content": a_text})

    return messages


def format_dailydialog_example(
    dialog_turns: List[str],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Convert a DailyDialog dialogue into alternating user/assistant messages."""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]

    for i, utt in enumerate(dialog_turns):
        text = utt.strip() if isinstance(utt, str) else ""
        if not text:
            continue
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": text})

    return messages


def format_prosocial_dialog_example(
    example: Dict[str, Any],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Convert an allenai/prosocial-dialog row into user/assistant messages."""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages: List[Dict[str, str]] = [{"role": "system", "content": sys_prompt}]

    context = example.get("context")
    if isinstance(context, str):
        ctx_str = context.strip()
        if ctx_str:
            messages.append({"role": "user", "content": ctx_str})
    elif isinstance(context, list):
        for i, turn in enumerate(context):
            turn_str = turn.strip() if isinstance(turn, str) else ""
            if turn_str:
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": turn_str})

    resp = example.get("response")
    if isinstance(resp, str) and resp.strip():
        # Ensure the last message before response was user; if not, add user turn
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": "How would you handle this situation?"})
        messages.append({"role": "assistant", "content": resp.strip()})

    return messages


def format_oasst1_example(
    prompt: str,
    response: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Convert an OpenAssistant prompt-response pair into chat messages."""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": response.strip()},
    ]


def extract_raw_texts_for_tokenizer(all_dialogues: List[List[Dict[str, str]]]) -> List[str]:
    """Extract flat text utterances from a collection of formatted dialogue messages."""
    texts = [DEFAULT_SYSTEM_PROMPT]
    for msgs in all_dialogues:
        for m in msgs:
            content = m.get("content", "").strip()
            if content:
                texts.append(content)
    return texts
