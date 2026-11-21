"""Tests for the training pipeline: loss masking, optimisation, eval, checkpoints."""

import json

import pytest
import torch
from torch.utils.data import DataLoader

from hibari.data.collator import DataCollatorForCausalLM, LengthGroupedBatchSampler
from hibari.data.dataset import ConversationDataset
from hibari.data.formatting import DEFAULT_SYSTEM_PROMPT
from hibari.data.tokenizer import HibariTokenizer
from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM
from hibari.training.checkpoint import load_checkpoint, save_checkpoint
from hibari.training.optimizer import configure_optimizers
from hibari.training.scheduler import get_cosine_schedule_with_warmup
from hibari.training.trainer import Trainer, TrainingArgs, evaluate_loss

CONVS = [
    [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How are you?"},
    ],
    [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "what is your name"},
        {"role": "assistant", "content": "I'm Hibari Koto."},
        {"role": "user", "content": "nice"},
        {"role": "assistant", "content": "Thank you!"},
    ],
] * 8


@pytest.fixture(scope="module")
def tok():
    t = HibariTokenizer()
    t.train_from_iterator([m["content"] for c in CONVS for m in c] * 3, vocab_size=300, min_frequency=1)
    return t


def _model(tok, seed=0):
    torch.manual_seed(seed)
    return HibariKotoLM(ModelConfig(vocab_size=tok.vocab_size, max_seq_len=128, d_model=64, n_layers=2, n_heads=2, d_ff=128))


def _loader(ds, tok, bs=4):
    return DataLoader(ds, batch_size=bs, collate_fn=DataCollatorForCausalLM(tok.pad_id))


def test_loss_mask_only_supervises_assistant_tokens(tok):
    ds = ConversationDataset(CONVS[:2], tok, max_seq_len=128)
    ex = ds[1]
    inp, tgt = ex["input_ids"].tolist(), ex["targets"].tolist()
    supervised = [t for t in tgt if t != -100]
    text = tok.decode(supervised)
    assert "I'm Hibari Koto." in text and "Thank you!" in text
    assert "what is your name" not in text and "nice" not in text
    assert DEFAULT_SYSTEM_PROMPT not in text
    assert supervised.count(tok.eos_id) == 2  # each reply ends with a supervised <|eos|>
    assert len(inp) == len(tgt)


def test_dataset_accepts_source_records(tok):
    recs = [{"source": "curated", "messages": c} for c in CONVS[:2]]
    ds = ConversationDataset(recs, tok, max_seq_len=128)
    assert len(ds) == 2 and ds.sources == ["curated", "curated"]


def test_collator_pads_and_masks(tok):
    ds = ConversationDataset(CONVS[:2], tok, max_seq_len=128)
    b = DataCollatorForCausalLM(tok.pad_id)([ds[0], ds[1]])
    assert b["input_ids"].shape == b["targets"].shape == b["attention_mask"].shape
    short = ds[0]["length"]
    assert (b["targets"][0, short:] == -100).all()
    assert (b["input_ids"][0, short:] == tok.pad_id).all()


def test_length_grouped_sampler_covers_all_and_is_deterministic():
    lengths = [5, 50, 7, 49, 6, 51, 8, 48]
    s1 = list(LengthGroupedBatchSampler(lengths, batch_size=2, group=4, seed=1))
    s2 = list(LengthGroupedBatchSampler(lengths, batch_size=2, group=4, seed=1))
    assert s1 == s2
    assert sorted(i for b in s1 for i in b) == list(range(8))


def test_backward_and_optimizer_step_reduce_loss(tok):
    model = _model(tok)
    ds = ConversationDataset(CONVS, tok, max_seq_len=128)
    batch = DataCollatorForCausalLM(tok.pad_id)([ds[i] for i in range(4)])
    opt = configure_optimizers(model, learning_rate=3e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        _, loss, _ = model(batch["input_ids"], targets=batch["targets"])
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_weight_decay_groups(tok):
    opt = configure_optimizers(_model(tok), weight_decay=0.1)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])


def test_cosine_schedule_warmup_and_floor(tok):
    opt = configure_optimizers(_model(tok), learning_rate=1.0)
    sch = get_cosine_schedule_with_warmup(opt, 10, 100, min_lr_ratio=0.1)
    lrs = []
    for _ in range(120):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert lrs[0] == 0.0 and abs(lrs[10] - 1.0) < 1e-6 and abs(lrs[-1] - 0.1) < 1e-6


def test_evaluate_loss_is_token_weighted(tok):
    model = _model(tok).eval()
    ds = ConversationDataset(CONVS[:3], tok, max_seq_len=128)
    m1 = evaluate_loss(model, _loader(ds, tok, bs=1), torch.device("cpu"))
    m3 = evaluate_loss(model, _loader(ds, tok, bs=3), torch.device("cpu"))
    assert abs(m1["eval_loss"] - m3["eval_loss"]) < 1e-4  # independent of batching


def test_trainer_runs_saves_best_and_history(tok, tmp_path):
    model = _model(tok)
    ds = ConversationDataset(CONVS, tok, max_seq_len=128)
    args = TrainingArgs(learning_rate=3e-3, warmup_steps=2, max_steps=20, batch_size=4,
                        eval_every=10, checkpoint_every=10, log_every=5, output_dir=str(tmp_path))
    opt = configure_optimizers(model, learning_rate=args.learning_rate)
    sch = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.max_steps)
    out = Trainer(model, opt, sch, _loader(ds, tok), _loader(ds, tok), args).train()
    assert out["best"]["final_step"] == 20
    for f in ["best_model.pt", "final_model.pt", "last_checkpoint.pt", "training_history.json"]:
        assert (tmp_path / f).exists()
    hist = json.loads((tmp_path / "training_history.json").read_text())
    train_losses = [h["train_loss"] for h in hist["history"] if "train_loss" in h]
    assert train_losses[-1] < train_losses[0]
    # periodic checkpoint stores the real recent loss (original bug: always 0.0)
    assert torch.load(tmp_path / "last_checkpoint.pt", weights_only=False)["loss"] > 0.0


def test_trainer_refuses_to_clobber_best_checkpoint(tok, tmp_path):
    (tmp_path / "best_model.pt").write_bytes(b"x")
    model = _model(tok)
    with pytest.raises(FileExistsError):
        Trainer(model, configure_optimizers(model), None, [], None, TrainingArgs(output_dir=str(tmp_path)))


def test_checkpoint_roundtrip_exact(tok, tmp_path):
    model = _model(tok).eval()
    save_checkpoint(tmp_path / "c.pt", model, step=7, loss=1.5)
    loaded = load_checkpoint(tmp_path / "c.pt")
    assert loaded["step"] == 7 and loaded["config"] == model.config
    ids = torch.randint(7, tok.vocab_size, (1, 10))
    with torch.no_grad():
        assert torch.equal(model(ids)[0], loaded["model"].eval()(ids)[0])
    # weight tying survives the round-trip
    m = loaded["model"]
    assert m.lm_head.weight.data_ptr() == m.embed_tokens.embedding.weight.data_ptr()


def test_checkpoint_with_run_metadata_loads_weights_only(tok, tmp_path):
    """Regression: run_metadata containing torch.__version__ must stay loadable."""
    model = _model(tok)
    save_checkpoint(tmp_path / "m.pt", model, extra={"run_metadata": {"torch": torch.__version__, "n": 1}})
    assert load_checkpoint(tmp_path / "m.pt")["step"] == 0
