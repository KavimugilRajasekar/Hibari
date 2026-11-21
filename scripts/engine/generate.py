"""Single-prompt generation and interactive multi-turn chat with Hibari Koto.

Examples:
    # interactive chat (commands: /reset, /exit)
    python hibari/scripts/generate.py

    # single prompt, deterministic
    python hibari/scripts/generate.py --prompt "Hi! How are you?" --temperature 0

    # raw model only (disable the application safety layer)
    python hibari/scripts/generate.py --no-safety
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.conversation import HibariChatSession
from hibari.training.checkpoint import load_checkpoint

DEFAULT_CKPT = "models/Koto/checkpoints/koto_start/best_model.pt"


def build_session(args) -> HibariChatSession:
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_checkpoint(ckpt, device=device)
    tokenizer = HibariTokenizer.load(args.tokenizer_dir)
    if tokenizer.vocab_size != loaded["config"].vocab_size:
        raise SystemExit("Tokenizer and checkpoint vocabulary sizes differ.")
    return HibariChatSession(
        model=loaded["model"].eval(), tokenizer=tokenizer, system_prompt=args.system,
        device=device, enable_safety=not args.no_safety,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with Hibari Koto")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer-dir", default="models/Koto/tokenizer")
    ap.add_argument("--prompt", default=None, help="Single prompt; prints one reply and exits")
    ap.add_argument("--system", default=None, help="Override the system prompt")
    ap.add_argument("--temperature", type=float, default=0.5, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-safety", action="store_true", help="Disable the deterministic safety layer")
    args = ap.parse_args()

    session = build_session(args)
    gen = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
               top_p=args.top_p, repetition_penalty=args.repetition_penalty, seed=args.seed)

    if args.prompt is not None:
        print(session.reply(args.prompt, **gen))
        return

    n = session.model.count_parameters()["total"]
    print(f"Hibari Koto ({n:,} parameters). Commands: /reset, /exit")
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in ("/exit", "/quit", "exit", "quit"):
            break
        if user == "/reset":
            session.reset()
            print("(conversation cleared)")
            continue
        try:
            print(f"Hibari: {session.reply(user, **gen)}")
        except Exception as e:  # keep the chat alive on unexpected errors
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
