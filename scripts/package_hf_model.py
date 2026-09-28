from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_tree_subset(src: Path, dst: Path, files: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(src / name, dst / name)


def package_hf_repo(checkpoint: str | Path, reference_store: str | Path, output_dir: str | Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(checkpoint); reference_store = Path(reference_store); out = Path(output_dir)
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    if not reference_store.exists(): raise FileNotFoundError(reference_store)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    required = ["model.pt", "amanah_config.json"]
    for name in required:
        path = checkpoint / name
        if not path.exists(): raise FileNotFoundError(path)
    for path in checkpoint.iterdir():
        if path.is_file(): shutil.copy2(path, out / path.name)
    shutil.copy2(reference_store, out / "reference_store.json")
    shutil.copy2(root / "deployment" / "hf_handler.py", out / "handler.py")
    _copy_tree_subset(root / "amanah_engine", out / "amanah_engine", ["__init__.py","classifier.py","references.py","schemas.py","scoring.py","semantic_rules.py","service.py"])
    _copy_tree_subset(root / "training", out / "training", ["__init__.py", "modeling.py"])
    _copy_tree_subset(root / "data", out / "data", ["__init__.py", "models.py"])
    requirements = "\n".join(["pydantic>=2.8,<3","torch>=2.3,<3","transformers>=4.45,<5","huggingface_hub>=0.35,<1","sentencepiece>=0.2,<1","protobuf>=5,<7"]) + "\n"
    (out / "requirements.txt").write_text(requirements, encoding="utf-8")
    model_card = root / "MODEL_CARD.md"
    if model_card.exists(): shutil.copy2(model_card, out / "README.md")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Package a trained AMANAH checkpoint for a Hugging Face custom endpoint")
    p.add_argument("--checkpoint", required=True); p.add_argument("--reference-store", required=True); p.add_argument("--output", default="artifacts/hf_model_repo")
    args = p.parse_args(); print(package_hf_repo(args.checkpoint, args.reference_store, args.output))


if __name__ == "__main__": main()
