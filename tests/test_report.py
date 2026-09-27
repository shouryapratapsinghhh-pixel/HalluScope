import numpy as np
import pandas as pd

from halluscope.report import END, START, build, inject_readme


def _fake_model(root, name, beyond, robust=True):
    d = root / name
    d.mkdir()
    pd.DataFrame({
        "comparison": ["activations+questions vs questions alone", "probe vs other model's activations",
                       "probe vs best output-confidence baseline", "probe+output vs output alone"],
        "mean_diff": [beyond, 0.01, 0.02, 0.05], "std_diff": [0.01, 0.01, 0.02, 0.01],
        "min_diff": [beyond - 0.02, -0.01, 0.0, 0.03],
        "significant_seeds": ["5/5", "0/5", "1/5", "5/5"], "robust": [robust, False, False, True],
    }).to_csv(d / "robustness.csv", index=False)
    pd.DataFrame({"seed": [0, 1], "best_layer": [10, 12], "probe_auroc": [0.7, 0.8],
                  "spearman_probe_vs_output": [0.3, 0.5]}).to_csv(d / "robustness_per_seed.csv", index=False)
    cache = root / f"{name}.npz"
    np.savez(cache, y_wrong=np.array([1, 1, 0, 1]))  # accuracy 0.25
    return {"name": name, "family": "F", "params_m": 100, "report": str(d), "cache": str(cache)}


def _fake_transfer(root, name, aurocs):
    d = root / name
    d.mkdir()
    # layer 1 has the best QA validation -> it is the headline, not layer 2's higher transfer
    pd.DataFrame({"layer": [1, 2, 3], "qa_val_auroc": [0.9, 0.6, 0.5],
                  "transfer_auroc": aurocs + [0.05]}).to_csv(d / "transfer.csv", index=False)
    return str(d)


def test_build_uses_saved_numbers_and_qa_chosen_layer(tmp_path):
    cfg = {"models": [_fake_model(tmp_path, "m1", 0.123)],
           "transfer": [{"name": "cities", "dirs": [_fake_transfer(tmp_path, "t1", [0.70, 0.99])]}]}
    md = build(cfg, out_dir=tmp_path / "out", figure=tmp_path / "scale.png")
    assert "+0.123 (5/5) ✅" in md
    assert "25.0%" in md  # accuracy read from the cache
    assert "| cities | 0.700 | 0.990 | L3 = 0.050 |" in md  # QA-chosen headline; best + worst descriptive
    assert (tmp_path / "scale.png").stat().st_size > 0
    assert (tmp_path / "out" / "scale_table.csv").exists()


def test_inject_readme_replaces_only_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{START}\nold\n{END}\n\nfooter\n")
    inject_readme(readme, f"{START}\nnew\n{END}")
    text = readme.read_text()
    assert "new" in text and "old" not in text and "# Title" in text and "footer" in text


def test_inject_readme_appends_when_no_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n")
    inject_readme(readme, f"{START}\nx\n{END}")
    assert START in readme.read_text()


def _fake_cost(root):
    d = root / "sc"
    d.mkdir()
    pd.DataFrame({"method": ["probe (prompt only)", "neg_min_logprob (greedy answer)",
                             "self-consistency (greedy + 10 samples)"],
                  "auroc": [0.80, 0.78, 0.75], "ci_low": [0.76, 0.74, 0.70], "ci_high": [0.84, 0.82, 0.79],
                  "forward_steps": [1.0, 7.5, 68.0], "ms_per_question": [99, 431, 3660],
                  "cost_vs_probe": [1.0, 7.5, 68.0]}).to_csv(d / "cost_vs_accuracy.csv", index=False)
    (d / "diffs.json").write_text(
        '{"probe - self_consistency": {"diff": 0.05, "ci_low": -0.003, "ci_high": 0.103}}')
    return {"name": "bench", "dir": str(d)}


def test_cost_section_and_key_findings_are_generated(tmp_path):
    cfg = {"models": [_fake_model(tmp_path, "m1", 0.123), _fake_model(tmp_path, "m2", 0.2)],
           "transfer": [{"name": "Negated city facts", "dirs": [_fake_transfer(tmp_path, "neg", [0.5, 0.6])]}],
           "cost": _fake_cost(tmp_path)}
    md = build(cfg, out_dir=tmp_path / "out", figure=tmp_path / "scale.png")
    assert "### Key findings" in md and "### Cost: probe vs self-consistency" in md
    assert "+0.12 to +0.20" in md  # range computed from the two models
    assert "68× fewer" in md  # 68.0 / 1.0 forward passes
    assert "matched (difference not significant)" in md  # wording follows the CI
    assert "layer 3 inverts to 0.050" in md  # most-inverted layer surfaced
    assert "(not significant)" in md  # CI [-0.003, +0.103] includes 0
