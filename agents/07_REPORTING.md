# 07 — Reporting to the Owner

The owner reads the final message, not your logs. Lead with the outcome.

## Structure
1. **Outcome** — one or two sentences: what works now, what does not.
2. **Evidence** — the commands run and their results (tests passed, validation errors,
   pass rates with seeds). Numbers only from runs in this session.
3. **Changes** — files changed per repository, one line each with the reason.
4. **Failures and limits** — concrete examples (input → output), regressions, latency.
5. **Needs training** — what, why decoding cannot fix it, which prepared config.
6. **Owner actions** — exact commands for training / pushing / releasing, which you did not run.

## Tone
- Plain, specific, no hype. "Pass rate 0.590 → 0.660 (3 seeds)", not "massively improved".
- State uncertainty plainly; if something was skipped or not verified, say so.
- Never claim the model "understands like a human" or that a suite pass proves quality.

## Pushing
The owner pushes. End with the commands, e.g.:
```bash
cd ~/Desktop/Hibari/models/Koto && git push origin <branch>
cd ~/Desktop/Hibari && git push origin <branch>
```
(Koto first, so the submodule commit exists on GitHub before the Hibari pointer.)
