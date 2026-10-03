import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

# Pin code and weights together: these releases execute Cloudflare's custom runtime.
CLEF_REVISIONS = {
    "clef": "2f3de3dd85f379784083b0814d997ab627200f0c",
    "clef-flash": "17f0b0ad64efb65d273590632833508766b2aae6",
}

type Predictor = Callable[
    [str | dict[str, Any] | list[Any], dict[str, dict[str, Any]], list[Image.Image]],
    dict[str, Any],
]


def load_predictor(model_name: str, device: str | None = None) -> Predictor:
    if model_name == "laya":
        import laya

        agent = laya.load("convaiinnovations/laya", device=device)

        def predict_laya(state, questions, images):
            return agent.predict(state, questions)

        return predict_laya

    if model_name not in CLEF_REVISIONS:
        raise ValueError("SYSTEMONE_MODEL must be laya, clef, or clef-flash")

    import torch
    from huggingface_hub import snapshot_download

    revision = CLEF_REVISIONS[model_name]
    path = Path(snapshot_download(f"Cloudflare/{model_name}", revision=revision))
    module_name = f"_systemone_clef_{revision}"
    spec = importlib.util.spec_from_file_location(module_name, path / "joint_schema_model.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the Clef release runtime")
    runtime = importlib.util.module_from_spec(spec)
    # Dataclasses in the release resolve annotations through sys.modules.
    sys.modules[module_name] = runtime
    try:
        spec.loader.exec_module(runtime)
    except BaseException:
        del sys.modules[module_name]
        raise
    model, processor = runtime.load_release_model(
        path, device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    def predict_clef(state, questions, images):
        # Laya accepts a list shorthand; Clef requires named options.
        for question in questions.values():
            if question["type"] == "choice" and isinstance(question["criteria"], list):
                question["criteria"] = dict.fromkeys(question["criteria"])
        return runtime.systemone(
            model,
            processor,
            {"model": model_name, "state": state, "questions": questions, "images": images},
            max_length=65_536,
        )

    return predict_clef
