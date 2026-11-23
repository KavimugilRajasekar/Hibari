"""Release profiles, provenance and Modelfile generation (hibari.release).

P1 a consistent release validates with no errors · P2 a wrong checkpoint
hash is an error · P3 a GGUF without provenance (the shipped 'koto_v4'
files) is an error · P4 a GGUF from other weights is an error · P5 a
hand-edited Modelfile is an error · P6 the exporter refuses to replace a
file with different content · P7 unknown inference keys are rejected ·
P8 the Modelfile carries the profile's settings · P9 engine config templates
match the dataclasses they document.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from hibari.data.tokenizer import HibariTokenizer
from hibari.export.gguf_export import export_to_gguf
from hibari.model.config import ModelConfig
from hibari.model.llm import HibariKotoLM
from hibari.release import InferenceSettings, ModelProfile, errors, render_modelfile, sha256_file, tokenizer_sha256, validate_profile
from hibari.training.checkpoint import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_script", ROOT / "scripts" / "engine" / "export.py")
export_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_script)

CORPUS = ["hello world", "how are you", "I am fine thanks", "what is your name"] * 4
SYSTEM = "You are Testy, a tiny test model."


def make_repo(tmp: Path, version="0.1.1") -> Path:
    tok = HibariTokenizer()
    tok.train_from_iterator(CORPUS, vocab_size=280, min_frequency=1)
    tok.save(tmp / "tokenizer")
    cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=64, d_model=32, n_layers=1, n_heads=2, d_ff=64)
    (tmp / "configs").mkdir()
    cfg.to_yaml(tmp / "configs" / "model.yaml")
    torch.manual_seed(0)
    save_checkpoint(tmp / "ckpt.pt", HibariKotoLM(cfg), step=7, extra={"run_metadata": {"model_version": "0.1.0"}})
    (tmp / "configs" / "inference.yaml").write_text(yaml.safe_dump(
        {"max_new_tokens": 20, "temperature": 0.5, "top_k": 10, "top_p": 0.9, "repetition_penalty": 1.1, "num_ctx": 64}))
    prof = {
        "model": "testy", "version": version, "weights_version": "0.1.0", "training_status": "NOT_REQUIRED",
        "identity": {"name": "Testy", "terms": ["testy"]}, "system_prompt": SYSTEM, "training_system_prompt": SYSTEM,
        "model_config": "configs/model.yaml", "inference_config": "configs/inference.yaml",
        "checkpoint": "ckpt.pt", "checkpoint_sha256": sha256_file(tmp / "ckpt.pt"),
        "tokenizer_dir": "tokenizer", "tokenizer_sha256": tokenizer_sha256(tmp / "tokenizer"),
        "gguf": {"dir": "gguf", "files": {"f32": {"file": "T-F32.gguf"}, "f16": {"file": "T-F16.gguf"}}},
    }
    p = tmp / "configs" / "release.yaml"
    # record hashes inline the way export --record writes them
    text = yaml.safe_dump(prof, sort_keys=False)
    text = text.replace("  files:\n    f32:\n      file: T-F32.gguf\n    f16:\n      file: T-F16.gguf\n",
                        "  files:\n    f32: {file: T-F32.gguf}\n    f16: {file: T-F16.gguf}\n")
    p.write_text(text)
    return p


@pytest.fixture
def released(tmp_path):
    p = make_repo(tmp_path)
    prof = ModelProfile.load(p, root=tmp_path)
    res = export_script.export_profile(prof, ["f32", "f16"], None, force=False)
    export_script.record_hashes(prof, res)
    return ModelProfile.load(p, root=tmp_path)


def test_p1_consistent_release_validates(released):
    found = validate_profile(released)
    assert not errors(found), [str(f) for f in errors(found)]
    assert {f.code for f in found} >= {"gguf:f32:provenance", "gguf:f16:vocab", "modelfile", "sha256:checkpoint"}
    assert all(g.sha256 for g in released.gguf_files)


def test_p2_wrong_checkpoint_hash(released):
    released.checkpoint_sha256 = "0" * 64
    assert "sha256:checkpoint" in {f.code for f in errors(validate_profile(released, check_gguf=False))}


def test_p3_gguf_without_provenance(released, tmp_path):
    from hibari.training.checkpoint import load_checkpoint
    m = load_checkpoint(released.resolve(released.checkpoint))["model"]
    tok = HibariTokenizer.load(released.resolve(released.tokenizer_dir))
    path = released.gguf_path("f32")
    path.unlink()
    export_to_gguf(m, tok, path, "f32", system_prompt=SYSTEM)  # legacy export: no hibari.* keys
    released.gguf_files[0].sha256 = None
    codes = {f.code for f in errors(validate_profile(released))}
    assert "gguf:f32:provenance" in codes


def test_p4_gguf_from_other_weights(released):
    other = HibariKotoLM(ModelConfig.from_yaml(released.resolve(released.model_config)))
    tok = HibariTokenizer.load(released.resolve(released.tokenizer_dir))
    path = released.gguf_path("f16")
    path.unlink()
    prov = dict(released.provenance(), **{"hibari.checkpoint.sha256": "f" * 64})
    export_to_gguf(other, tok, path, "f16", system_prompt=SYSTEM, provenance=prov)
    released.gguf_files[1].sha256 = None
    assert "gguf:f16:provenance" in {f.code for f in errors(validate_profile(released))}


def test_p5_hand_edited_modelfile(released):
    mf = released.resolve(released.gguf_dir) / "Modelfile"
    mf.write_text(mf.read_text().replace("temperature 0.5", "temperature 0.9"))
    assert "modelfile" in {f.code for f in errors(validate_profile(released))}


def test_p6_export_refuses_to_replace_different_file(released):
    path = released.gguf_path("f32")
    path.write_bytes(b"not the same")
    with pytest.raises(export_script.ExportError):
        export_script.export_profile(released, ["f32"], None, force=False)
    assert path.read_bytes() == b"not the same"
    # identical content is kept without error
    export_script.export_profile(released, ["f16"], None, force=False)


def test_p7_unknown_inference_key_rejected():
    with pytest.raises(ValueError):
        InferenceSettings.from_dict({"temprature": 0.5})


def test_p8_modelfile_from_profile(released):
    text = render_modelfile(released, "T-F32.gguf")
    assert "FROM ./T-F32.gguf" in text and f'SYSTEM """{SYSTEM}"""' in text
    assert "PARAMETER temperature 0.5" in text and "PARAMETER num_ctx 64" in text and "PARAMETER num_predict 20" in text
    assert 'PARAMETER stop "<|eos|>"' in text


def test_p9_engine_config_templates_match_code():
    from hibari.training.trainer import TrainingArgs
    tdir = ROOT / "hibari" / "configs"
    model = yaml.safe_load((tdir / "model" / "base.yaml").read_text())
    assert set(model) <= set(ModelConfig.__dataclass_fields__), set(model) - set(ModelConfig.__dataclass_fields__)
    ModelConfig.from_dict(model).head_dim  # divisible
    for f in ("base.yaml", "sft.yaml"):
        train = yaml.safe_load((tdir / "training" / f).read_text())
        extra = {"experiment", "model_version", "init_checkpoint", "data_cache_dir", "val_examples"}
        assert set(train) - extra <= set(TrainingArgs.__dataclass_fields__), set(train) - extra - set(TrainingArgs.__dataclass_fields__)
    InferenceSettings.from_dict(yaml.safe_load((tdir / "inference" / "base.yaml").read_text()))
    rel = yaml.safe_load((tdir / "release" / "template.yaml").read_text())
    for k in ("model", "version", "weights_version", "system_prompt", "checkpoint", "tokenizer_dir", "inference_config"):
        assert k in rel



def test_checkpoint_saved_before_a_new_config_field_still_validates(tmp_path):
    """A checkpoint written before resid_dropout existed (all Koto weights up to 1.3.0) must still
    match its model config; engine 0.4.1 reported 'checkpoint vs config differ' without the fix."""
    p = make_repo(tmp_path)
    ck = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    ck["config"].pop("resid_dropout")
    torch.save(ck, tmp_path / "ckpt.pt")
    text = p.read_text()
    old_sha = yaml.safe_load(text)["checkpoint_sha256"]
    p.write_text(text.replace(old_sha, sha256_file(tmp_path / "ckpt.pt")))
    found = validate_profile(ModelProfile.load(p, root=tmp_path), check_gguf=False)
    assert "checkpoint:config" not in {f.code for f in errors(found)}, [str(f) for f in errors(found)]


def test_unknown_modelfile_variant_is_a_finding_not_a_crash(released):
    # next(...) over the GGUF list raised a bare StopIteration for a variant the profile does not list
    found = validate_profile(released, check_hashes=False, modelfile_variant="q3_nope")
    assert any(f.code == "modelfile" and f.level == "error" and "q3_nope" in f.message for f in found), [str(f) for f in found]


def test_export_with_no_variant_produced_is_an_export_error(tmp_path, monkeypatch):
    # every requested variant skipped (no llama-quantize): next(...) raised a bare StopIteration
    p = make_repo(tmp_path)
    prof = ModelProfile.load(p, root=tmp_path)
    monkeypatch.setattr(export_script.shutil, "which", lambda _: None)
    prof.gguf_files.append(type(prof.gguf_files[0])(variant="q8_0", file="T-Q8_0.gguf"))
    with pytest.raises(export_script.ExportError, match="no GGUF"):
        export_script.export_profile(prof, ["q8_0"], None, force=False)
    assert not (tmp_path / "gguf" / "Modelfile").exists()
