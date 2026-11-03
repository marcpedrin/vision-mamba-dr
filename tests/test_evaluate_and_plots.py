"""End to end on fake data: train -> evaluate (both datasets) -> all 5 figures."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic import make_synthetic_eyepacs, make_synthetic_messidor
from eval import evaluate, plots
from train import train

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def posix(path):
    """Forward slashes so its safe in a yaml override."""
    return str(path).replace(os.sep, "/")


def test_evaluate_then_plot(tmp_path):
    #100 fake EyePACS, 10 + 1 ungradable Messidor
    image_dir, labels_csv = make_synthetic_eyepacs(str(tmp_path / "eyepacs"),
                                                   num_patients=50)
    m_dir, m_csv = make_synthetic_messidor(str(tmp_path / "messidor"), num_images=10)

    #shared by all 3 steps
    run_dir = tmp_path / "run"
    common = [
        "--config", CONFIG_PATH,
        "--override", f"data.eyepacs_image_dir={posix(image_dir)}",
        "--override", f"data.eyepacs_labels_csv={posix(labels_csv)}",
        "--override", f"data.messidor_image_dir={posix(m_dir)}",
        "--override", f"data.messidor_labels_csv={posix(m_csv)}",
        "--override", "data.num_workers=0",
        "--override", f"train.output_dir={posix(run_dir)}",
        "--override", "model.depth=2",
    ]

    #1. train
    summary = train.main(common + ["--smoke-test", "--device", "cpu"])
    smoke_dir = summary["output_dir"]

    #2. evaluate best.pt
    eval_paths = [
        "--override", f"eval.checkpoint={posix(os.path.join(smoke_dir, 'best.pt'))}",
        "--override", f"eval.results_json={posix(os.path.join(smoke_dir, 'res.json'))}",
        "--override", f"eval.figures_dir={posix(tmp_path / 'figures')}",
    ]
    results = evaluate.main(common + eval_paths + ["--device", "cpu"])

    assert set(results.keys()) == {"eyepacs_test", "messidor2"}
    for metrics in results.values():
        for key in ("accuracy", "qwk", "macro_f1", "confusion_matrix", "loss"):
            assert key in metrics
    #ungradable one skipped
    assert results["messidor2"]["num_images"] == 10

    with open(os.path.join(smoke_dir, "res.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert "eyepacs_test" in saved["results"]
    assert os.path.exists(os.path.join(smoke_dir, "predictions_eyepacs_test.csv"))
    assert os.path.exists(os.path.join(smoke_dir, "predictions_messidor2.csv"))

    #3. figures
    plot_args = common + eval_paths + ["--override",
                                       f"train.output_dir={posix(smoke_dir)}"]
    written = plots.main(plot_args)
    names = sorted(os.path.basename(p) for p in written)
    print("Figures:", names)
    assert names == [
        "confusion_eyepacs_test.png",
        "confusion_messidor2.png",
        "examples_eyepacs_test.png",
        "examples_messidor2.png",
        "training_curves.png",
    ]


if __name__ == "__main__":
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        test_evaluate_then_plot(pathlib.Path(folder))
    print("OK")
