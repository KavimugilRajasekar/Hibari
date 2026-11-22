"""Single-prompt generation and interactive multi-turn chat with a Hibari model.

Preferred: run from a release profile, which supplies the checkpoint, the
tokenizer, the system prompt and every inference setting (the same values
the evaluation and the Ollama Modelfile use)::

    python scripts/engine/generate.py --profile models/Koto/configs/release/koto_1.1.1.yaml
    python scripts/engine/generate.py --profile ... --prompt "Hi! How are you?" --temperature 0
    python scripts/engine/generate.py --profile ... --raw          # model only: no safety, router or reranking
    python scripts/engine/generate.py --profile ... --explain      # show the reply candidates and their scores

Commands in chat: /reset, /exit. Command-line decoding options override the profile.
"""

from __future__ import annotations

import argparse
import sys

import torch

from hibari.inference.engine import InferenceEngine

DEFAULT_PROFILE = "models/Koto/configs/release/koto_1.1.1.yaml"


def build(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint:  # legacy: checkpoint + tokenizer, engine defaults
        engine = InferenceEngine.from_checkpoint(args.checkpoint, args.tokenizer_dir, device)
        if engine.tokenizer.vocab_size != engine.model.config.vocab_size:
            raise SystemExit("Tokenizer and checkpoint vocabulary sizes differ.")
    else:
        engine = InferenceEngine.from_profile(args.profile, device)
    overrides = {}
    if args.raw:
        overrides = dict(enable_safety=False, route_tasks=False, normalize_input=False, num_candidates=1)
    elif args.no_safety:
        overrides = dict(enable_safety=False)
    if args.candidates is not None:
        overrides["num_candidates"] = args.candidates
    session = engine.create_chat_session(system_prompt=args.system, **overrides)
    gen = dict(engine.reply_settings() or dict(max_new_tokens=80, temperature=0.5, top_k=25, top_p=0.85, repetition_penalty=1.15))
    for key in ("max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty"):
        v = getattr(args, key)
        if v is not None:
            gen[key] = v
    gen["seed"] = args.seed
    return engine, session, gen


def show(session, explain: bool) -> None:
    if not explain:
        return
    info = session.last_reply_info or {}
    if info.get("analysis"):
        a = info["analysis"]
        print(f"  [intent={a['intent']} emotion={a['emotion']} hostile={a['hostile']} model_input={info.get('model_input')!r}]")
    for c in info.get("candidates", [])[:5]:
        print(f"  {c['score']:6.2f} {c['text'][:70]!r} {c['adjustments']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with a Hibari model")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--checkpoint", default=None, help="Legacy: checkpoint path instead of a profile")
    ap.add_argument("--tokenizer-dir", default="models/Koto/tokenizer")
    ap.add_argument("--prompt", default=None, help="Single prompt; prints one reply and exits")
    ap.add_argument("--system", default=None, help="Override the system prompt")
    ap.add_argument("--temperature", type=float, default=None, help="0 = greedy")
    ap.add_argument("--top-k", dest="top_k", type=int, default=None)
    ap.add_argument("--top-p", dest="top_p", type=float, default=None)
    ap.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=None)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    ap.add_argument("--candidates", type=int, default=None, help="Override num_candidates")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-safety", action="store_true", help="Disable the deterministic safety layer")
    ap.add_argument("--raw", action="store_true", help="Raw model: no safety, router, normalisation or reranking")
    ap.add_argument("--explain", action="store_true", help="Print the message analysis and the top candidates")
    args = ap.parse_args()

    engine, session, gen = build(args)
    name = engine.profile.identity.name if engine.profile else "Hibari"
    if args.prompt is not None:
        print(session.reply(args.prompt, **gen))
        show(session, args.explain)
        return

    n = engine.model.count_parameters()["total"]
    ver = f" {engine.profile.version}" if engine.profile else ""
    print(f"{name}{ver} ({n:,} parameters). Commands: /reset, /exit")
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
            print(f"{name}: {session.reply(user, **gen)}")
            show(session, args.explain)
        except Exception as e:  # keep the chat alive on unexpected errors
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
