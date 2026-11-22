# 04 — Testing and Evaluation

## Tests (must pass before you finish)
| Suite | Command | Count (2026-11-22) |
|---|---|---|
| Engine | `pytest -q` | 311 |
| Koto | `cd models/Koto && python -m pytest -q tests` | 69 |

Regression tests live next to their area: `tests/test_regressions.py` (0.2.0 review bugs),
`tests/test_conversational_layer.py` (analysis, normalisation, critic, session),
`tests/test_release.py` (profiles, provenance, Modelfile, config templates),
Koto `tests/test_release_and_behaviour.py` (leaks, behaviour data, configs, experiments).

## Behavioural evaluation
| Suite | File | Runner |
|---|---|---|
| Behaviour (Phase 9: basic, empathy, context, reformulation, writing, robustness, safety, identity) | `koto/evaluation/behaviour_suite.py` | `evaluation/evaluate_behaviour.py` |
| Upgrade (1.1.0: conversation, rephrase, grammar, preservation, dialogue) | `koto/evaluation/upgrade_suite.py` | `evaluation/evaluate_upgrade.py` |
| Legacy (58 cases) | `koto/evaluation/suite.py` | `evaluation/evaluate.py` |

Rules
- Compare versions on the **same** suite, seeds (0 1 2) and settings; report the mean and
  the seed agreement, not the best seed.
- Run `raw`, `baseline` and `release` pipelines so the effect of the application layer is visible.
- **Read the transcripts** in the `.md` report; quote failing examples in your report.
- Never edit suite prompts or checks to make a result look better. If a check is wrong,
  fix it in a separate, clearly reported change and re-run every compared version.
- Never add suite prompts to training data; the leak filter and tests enforce this.
- Tune on a separate development set (write it in the scratchpad, not in the suites).

Baseline to beat (1.1.1, `outputs/evaluation/koto_1.1.1_behaviour.md`): release pass rate
0.660 mean (empathy 0.70, context 0.40, writing 0.25, identity 0.33); raw model 0.481.
