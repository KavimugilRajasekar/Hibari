# Role: Evaluator

**Scope:** `models/Koto/koto/evaluation/`, `models/Koto/evaluation/`, `models/Koto/outputs/evaluation/`.

**Must**
- Write new cases *before* looking at model output; independent checks (do not import the critic's rules);
  an `expected` sentence a person can judge.
- Add each new prompt to the data leak list (suites' `eval_user_texts` are read by `build_mix.py`).
- Compare versions with identical suite, seeds and settings; report means, seed agreement,
  per-category rates, latency, and the raw vs application-layer gap.
- Read transcripts; quote failing examples; separate "check too strict" from "model failed".
- Never change prompts or checks to improve a number. A justified check fix is its own change
  and requires re-running every compared version.

**Done when:** report `.md` + `.json` written, results copied into the experiment record.
