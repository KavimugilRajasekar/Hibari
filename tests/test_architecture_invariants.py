"""Architecture invariants and previously untested runtime paths (2026-11-23 coverage pass).

Every test here exercises code that the suite did not reach before (measured with a line-coverage
run): the internal maths of the model, ``HibariKotoLM.generate``, rows leaving ``generate_batch``
early, ``evaluate_records``, ``InferenceEngine``, checkpoint loading, ``ModelConfig`` validation,
the quantization helpers and the NumPy GGUF reference runtime.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from hibari.data.tokenizer import HibariTokenizer
from hibari.inference.generation import generate_batch, generate_tokens
from hibari.model.config import ModelConfig
from hibari.model.embedding import apply_rotary_emb, precompute_rope_freqs
from hibari.model.llm import HibariKotoLM
from hibari.model.normalization import RMSNorm

CORPUS = ["hello world", "how are you", "I am fine thanks", "what is your name", "Hibari is here"] * 4

CONFIGS = [
    dict(vocab_size=96, max_seq_len=32, d_model=32, n_layers=2, n_heads=2, d_ff=64),
    dict(vocab_size=128, max_seq_len=48, d_model=48, n_layers=3, n_heads=4, d_ff=144, resid_dropout=0.1),
    dict(vocab_size=64, max_seq_len=16, d_model=16, n_layers=1, n_heads=1, d_ff=40, tie_embeddings=False),
]


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


@pytest.fixture(scope="module")
def tok():
    t = HibariTokenizer()
    t.train_from_iterator(CORPUS, vocab_size=300, min_frequency=1)
    return t


@pytest.fixture(scope="module", params=range(len(CONFIGS)), ids=["base", "resid_dropout", "untied_1layer"])
def arch(request):
    torch.manual_seed(request.param)
    return HibariKotoLM(ModelConfig(**CONFIGS[request.param])).eval()


# ───────────── structure ─────────────

def test_parameter_count_matches_formula_and_unique_tensors(arch):
    unique = {id(p): p.numel() for p in arch.parameters()}
    assert sum(unique.values()) == arch.count_parameters()["total"] == arch.config.estimate_parameters()


def test_tied_embeddings_share_storage_only_when_configured(arch):
    shared = arch.lm_head.weight.data_ptr() == arch.embed_tokens.embedding.weight.data_ptr()
    assert shared == arch.config.tie_embeddings


def test_weights_are_finite_and_projections_have_no_bias(arch):
    for name, p in arch.named_parameters():
        assert torch.isfinite(p).all(), name
        assert not name.endswith(".bias"), f"unexpected bias {name}"


# ───────────── forward-pass maths ─────────────

def test_causality_future_tokens_never_change_past_logits(arch):
    torch.manual_seed(1)
    T, V = arch.config.max_seq_len // 2, arch.config.vocab_size
    x = torch.randint(0, V, (1, T))
    base = _logits(arch(x))
    for j in (1, T // 2, T - 1):
        y = x.clone()
        y[0, j] = (y[0, j] + 1) % V
        out = _logits(arch(y))
        assert torch.equal(out[0, :j], base[0, :j]), f"position {j} leaked into the past"
        assert not torch.equal(out[0, j:], base[0, j:])


def test_kv_cache_incremental_decoding_equals_full_forward(arch):
    torch.manual_seed(2)
    T = arch.config.max_seq_len // 2
    x = torch.randint(0, arch.config.vocab_size, (1, T))
    full = _logits(arch(x))
    with torch.no_grad():
        _, _, caches = arch(x[:, :3], use_cache=True)
        steps = []
        for t in range(3, T):
            lg, _, caches = arch(x[:, t:t + 1], kv_caches=caches, use_cache=True)
            steps.append(lg[0, -1])
    assert torch.allclose(torch.stack(steps), full[0, 3:], atol=1e-5)


def test_batch_rows_are_independent(arch):
    torch.manual_seed(3)
    T = arch.config.max_seq_len // 2
    a, b = (torch.randint(0, arch.config.vocab_size, (1, T)) for _ in range(2))
    both = _logits(arch(torch.cat([a, b])))
    assert torch.allclose(both[0], _logits(arch(a))[0], atol=1e-5)
    assert torch.allclose(both[1], _logits(arch(b))[0], atol=1e-5)


def test_eval_mode_is_deterministic_even_with_dropout(arch):
    x = torch.randint(0, arch.config.vocab_size, (2, 8))
    with torch.no_grad():
        assert torch.equal(_logits(arch(x)), _logits(arch(x)))


def test_sequences_longer_than_the_context_extend_rope_without_changing_the_prefix(arch):
    torch.manual_seed(4)
    L = arch.config.max_seq_len
    x = torch.randint(0, arch.config.vocab_size, (1, L + 8))
    with torch.no_grad():
        prefix = _logits(arch(x[:, :L]))
        longer = _logits(arch(x))
    assert longer.shape[1] == L + 8
    assert torch.allclose(longer[0, :L], prefix[0], atol=1e-5)


def test_loss_ignores_masked_targets_and_matches_manual_cross_entropy(arch):
    torch.manual_seed(5)
    x = torch.randint(0, arch.config.vocab_size, (2, 10))
    targets = x.clone()
    targets[:, :4] = -100
    with torch.no_grad():
        logits, loss, _ = arch(x, targets=targets)
    manual = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
    assert torch.isclose(loss, manual, atol=1e-6)


def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(6)
    D = 16
    cos, sin = precompute_rope_freqs(head_dim=D, max_seq_len=64)
    q, k = torch.randn(1, 1, 1, D), torch.randn(1, 1, 1, D)

    def score(m, n):
        qm = apply_rotary_emb(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rotary_emb(k, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    for shift in (1, 7, 30):
        assert math.isclose(score(3, 1), score(3 + shift, 1 + shift), rel_tol=1e-4, abs_tol=1e-5)
    assert torch.allclose(apply_rotary_emb(q, cos[5:6], sin[5:6]).norm(), q.norm(), atol=1e-5)  # a rotation


def test_rmsnorm_is_scale_invariant():
    torch.manual_seed(7)
    n = RMSNorm(16, eps=1e-8)
    x = torch.randn(3, 16)
    assert torch.allclose(n(x), n(5.0 * x), atol=1e-5)


# ───────────── generation paths ─────────────

def test_model_generate_greedy_topk_and_eos():
    torch.manual_seed(8)
    m = HibariKotoLM(ModelConfig(**CONFIGS[0])).eval()
    x = torch.randint(0, 96, (1, 4))
    a = m.generate(x, max_new_tokens=6, temperature=0.9, top_k=1, top_p=1.0)   # top_k=1 is greedy
    b = m.generate(x, max_new_tokens=6, temperature=0.3, top_k=1, top_p=0.5)
    assert torch.equal(a, b) and a.shape[1] == 10
    first = int(a[0, 4])
    stopped = m.generate(x, max_new_tokens=6, temperature=0.9, top_k=1, top_p=1.0, eos_token_id=first)
    assert stopped.shape[1] <= 5
    sampled = m.generate(x, max_new_tokens=5, temperature=1.0, top_k=10, top_p=0.9)
    assert sampled.shape[1] == 9 and int(sampled.max()) < 96


def test_generate_batch_rows_that_stop_early_leave_the_batch_and_match_sequential(tok):
    """Exercises the KV-cache pruning branch (rows hitting a stop token mid-batch)."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=128, d_model=32, n_layers=2, n_heads=2, d_ff=64)
    model = HibariKotoLM(cfg).eval()
    prompt = tok.encode("hello")
    dec = dict(temperature=3.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
    for base in range(0, 400, 16):
        seeds = list(range(base, base + 16))
        res = generate_batch(model, tok, prompt, [dec] * 16, seeds, 60)
        reasons = [r for _, r in res]
        if "stop_token" in reasons and "max_new_tokens" in reasons:
            break
    else:
        pytest.fail("no seed range produced both early and full-length rows")
    seq = [generate_tokens(model, tok, "", prompt_ids=prompt, max_new_tokens=60, seed=s, **dec) for s in seeds]
    assert [ids for ids, _ in res] == seq


def test_generate_batch_argument_edge_cases(tok):
    torch.manual_seed(0)
    model = HibariKotoLM(ModelConfig(vocab_size=tok.vocab_size, max_seq_len=32, d_model=16, n_layers=1, n_heads=2, d_ff=32))
    with pytest.raises(ValueError):
        generate_batch(model, tok, [5], [{}], [1, 2], 4)
    assert generate_batch(model, tok, [5], [], [], 4) == []
    assert generate_batch(model, tok, [5], [{}, {}], [1, 2], 0) == [([], "max_new_tokens")] * 2


# ───────────── evaluation, engine, checkpoints ─────────────

def _records(n=6):
    from hibari.data.formatting import DEFAULT_SYSTEM_PROMPT
    out = []
    for i in range(n):
        out.append({"source": "a" if i % 2 else "b", "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": CORPUS[i % 5]},
            {"role": "assistant", "content": CORPUS[(i + 1) % 5]}]})
    return out


def test_evaluate_records_overall_and_per_source(tok):
    from hibari.evaluation.evaluator import evaluate_records
    torch.manual_seed(0)
    model = HibariKotoLM(ModelConfig(vocab_size=tok.vocab_size, max_seq_len=256, d_model=32, n_layers=1, n_heads=2, d_ff=64))
    res = evaluate_records(model, tok, _records(), batch_size=4)
    assert set(res) == {"all", "a", "b"}
    assert res["all"]["examples"] == 6 and res["a"]["examples"] == 3
    assert res["all"]["tokens"] == res["a"]["tokens"] + res["b"]["tokens"]
    weighted = (res["a"]["loss"] * res["a"]["tokens"] + res["b"]["loss"] * res["b"]["tokens"]) / res["all"]["tokens"]
    assert abs(weighted - res["all"]["loss"]) < 1e-3
    assert abs(math.exp(res["all"]["loss"]) - res["all"]["perplexity"]) / res["all"]["perplexity"] < 1e-3
    assert set(evaluate_records(model, tok, _records(), by_source=False)) == {"all"}


def test_checkpoint_roundtrip_training_state_and_missing_file(tmp_path):
    from hibari.training.checkpoint import load_checkpoint, save_checkpoint
    torch.manual_seed(0)
    m = HibariKotoLM(ModelConfig(**CONFIGS[1]))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    m(torch.randint(0, 128, (1, 4)), targets=torch.randint(0, 128, (1, 4)))[1].backward()
    opt.step()
    save_checkpoint(tmp_path / "c.pt", m, optimizer=opt, step=3, loss=1.5)
    loaded = load_checkpoint(tmp_path / "c.pt", load_training_state=True)
    assert loaded["optimizer_state_dict"]["state"] and loaded["model"].config == m.config
    x = torch.randint(0, 128, (1, 6))
    assert not loaded["model"].training    # resid_dropout 0.1: a train-mode model would be random
    assert torch.equal(_logits(loaded["model"](x)), _logits(loaded["model"](x)))
    m.eval()
    assert torch.equal(_logits(loaded["model"](x)), _logits(m(x)))
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_inference_engine_from_profile_and_checkpoint(tmp_path):
    from hibari.inference.engine import InferenceEngine
    from test_release import make_repo
    from hibari.release import ModelProfile
    p = make_repo(tmp_path)
    prof = ModelProfile.load(p, root=tmp_path)
    eng = InferenceEngine.from_profile(prof)
    assert isinstance(eng.generate("hello", max_new_tokens=5, temperature=0.0, seed=0), str)
    session = eng.create_chat_session()
    assert session is not None and eng.reply_settings()
    assert eng.rewrite("rephrase", "hello world").__class__.__name__ == "TaskResult"
    direct = InferenceEngine.from_checkpoint(tmp_path / "ckpt.pt", tmp_path / "tokenizer")
    assert direct.model.config == eng.model.config
    assert eng.profile is prof


# ───────────── config validation and quantization helpers ─────────────

@pytest.mark.parametrize("bad, msg", [
    (dict(d_model=30, n_heads=4), "divisible"),
    (dict(d_model=30, n_heads=10), "even for RoPE"),
    (dict(n_layers=-1), "n_layers"),
    (dict(vocab_size=True), "vocab_size"),
    (dict(norm_eps=0.0), "norm_eps"),
    (dict(rope_theta=-1.0), "rope_theta"),
    (dict(dropout=1.0), "dropout"),
])
def test_model_config_rejects_inconsistent_values(bad, msg):
    with pytest.raises(ValueError, match=msg):
        ModelConfig(**{**CONFIGS[0], **bad})


def test_model_config_strict_loading_and_head_dim():
    with pytest.raises(ValueError, match="Unknown ModelConfig keys"):
        ModelConfig.from_dict_strict({**CONFIGS[0], "typo_layers": 3})
    assert ModelConfig.from_dict({**CONFIGS[0], "ignored": 1}).head_dim == 16


def test_quantization_helpers():
    from hibari.export.quantization import calculate_quantization_stats, quantize_fp16
    sd = {"w": torch.ones(4, 4), "ids": torch.arange(3)}
    half = quantize_fp16(sd)
    assert half["w"].dtype == torch.float16 and half["ids"].dtype == torch.int64
    stats = calculate_quantization_stats(sd)
    assert stats["total_parameters"] == 19
    assert stats["total_megabytes"] == pytest.approx((16 * 4 + 3 * 8) / 2**20)


# ───────────── GGUF reference runtime (independent of llama.cpp) ─────────────

@pytest.mark.parametrize("qtype, atol", [("f32", 1e-4), ("f16", 5e-2)])
def test_gguf_reference_runtime_matches_pytorch(tok, tmp_path, qtype, atol):
    from hibari.export.gguf_export import export_to_gguf
    from hibari.export.gguf_reference import GGUFReferenceModel
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=64, d_model=32, n_layers=2, n_heads=2, d_ff=64)
    model = HibariKotoLM(cfg).eval()
    path = tmp_path / f"m_{qtype}.gguf"
    export_to_gguf(model, tokenizer=tok, output_path=path, quantization_type=qtype)
    ref = GGUFReferenceModel(path)
    ids = tok.encode("hello world how are you")
    with torch.no_grad():
        want = _logits(model(torch.tensor([ids])))[0].numpy()
    got = ref.forward(ids)
    assert got.shape == want.shape
    assert abs(got - want).max() < atol
    if qtype == "f32":
        greedy = generate_tokens(model, tok, "", prompt_ids=ids, max_new_tokens=8, temperature=0.0,
                                 repetition_penalty=1.0)
        from hibari.inference.generation import _stop_ids
        stops = _stop_ids(tok)
        assert ref.greedy(ids, 8, stop_ids=stops)[: len(greedy)] == greedy


# ───────────── scripts/engine/verify_architecture.py ─────────────

def _run_verifier(monkeypatch, args):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "engine" / "verify_architecture.py"
    spec = importlib.util.spec_from_file_location("verify_architecture_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr("sys.argv", ["verify_architecture.py"] + args)
    return mod.main()


def test_verify_architecture_passes_a_consistent_checkpoint_and_catches_a_foreign_gguf(tok, tmp_path, monkeypatch):
    import json
    from hibari.export.gguf_export import export_to_gguf
    from hibari.training.checkpoint import save_checkpoint
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=256, d_model=32, n_layers=2, n_heads=2, d_ff=64)
    torch.manual_seed(0)
    model = HibariKotoLM(cfg).eval()
    torch.manual_seed(1)
    other = HibariKotoLM(cfg).eval()
    tok.save(tmp_path / "tok")
    save_checkpoint(tmp_path / "m.pt", model, step=5)
    export_to_gguf(model, tokenizer=tok, output_path=tmp_path / "own-F32.gguf", quantization_type="f32")
    export_to_gguf(other, tokenizer=tok, output_path=tmp_path / "foreign-F32.gguf", quantization_type="f32")
    recs = tmp_path / "recs.json"
    recs.write_text(json.dumps(_records()))
    base = ["--checkpoint", str(tmp_path / "m.pt"), "--tokenizer-dir", str(tmp_path / "tok"), "--greedy-tokens", "6"]
    ok = _run_verifier(monkeypatch, base + ["--gguf", f"0={tmp_path / 'own-F32.gguf'}", "--records", str(recs),
                                            "--report", str(tmp_path / "ok.json")])
    rep = json.loads((tmp_path / "ok.json").read_text())["checkpoints"][0]
    assert ok == 0 and rep["ok"] and rep["structure"]["ok"] and rep["heldout"]["examples"] == 6
    assert all(m["ok"] for m in rep["maths"]) and rep["gguf"][0]["ok"]
    bad = _run_verifier(monkeypatch, base + ["--gguf", f"0={tmp_path / 'foreign-F32.gguf'}", "--report", str(tmp_path / "bad.json")])
    assert bad == 1 and not json.loads((tmp_path / "bad.json").read_text())["checkpoints"][0]["gguf"][0]["ok"]
