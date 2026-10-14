"""The package as a whole: its version and the names it exports."""

from importlib.metadata import version

import graphssl


def test_version_is_the_installed_one():
    assert graphssl.__version__ == version("graphssl")
    assert graphssl.__version__[0].isdigit()


def test_top_level_names_resolve():
    for name in (
        "BGRL",
        "GCNEncoder",
        "LogRegEvaluator",
        "RidgeEvaluator",
        "effective_rank",
        "extract_embeddings",
        "DINOTrainer",
    ):
        assert hasattr(graphssl, name), name
