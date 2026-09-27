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
    pd.DataFrame({"layer": [1, 2], "qa_val_auroc": [0.9, 0.6],
                  "transfer_auroc": aurocs}).to_csv(d / "transfer.csv", index=False)
    return str(d)


def test_build_uses_saved_numbers_and_qa_chosen_layer(tmp_path):
    cfg = {"models": [_fake_model(tmp_path, "m1", 0.123)],
           "transfer": [{"name": "cities", "dirs": [_fake_transfer(tmp_path, "t1", [0.70, 0.99])]}]}
    md = build(cfg, out_dir=tmp_path / "out", figure=tmp_path / "scale.png")
    assert "+0.123 (5/5) ✅" in md
    assert "25.0%" in md  # accuracy read from the cache
    assert "| cities | 0.700 | 0.990 |" in md  # headline = QA-chosen layer, best-any is descriptive
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
