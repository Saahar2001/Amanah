from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.build_dataset import build_faithful_samples
from data.models import AmanahSample, DriftLabel, ReferenceTranslation, Severity
from data.mutations import apply_condition_loss, apply_modality_shift, apply_negation_flip, apply_omission, apply_quantifier_change
from data.split_dataset import assert_no_ayah_leakage, group_split
from data.validators import validate_mutation

OPERATORS = [apply_negation_flip,apply_quantifier_change,apply_modality_shift,apply_omission,apply_condition_loss]
SEVERITY = {DriftLabel.NEGATION_FLIP: Severity.S3,DriftLabel.QUANTIFIER_CHANGE: Severity.S3,DriftLabel.MODALITY_SHIFT: Severity.S3,DriftLabel.CONDITION_LOSS: Severity.S3,DriftLabel.OMISSION: Severity.S2}


def _dump_jsonl(path: Path, rows: list[AmanahSample]) -> None:
    path.write_text("\n".join(r.model_dump_json() for r in rows) + ("\n" if rows else ""), encoding="utf-8")


def prepare_dataset(source_path: str | Path, output_dir: str | Path, *, seed: int = 42) -> dict:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8")); ayahs = payload.get("ayahs", payload) if isinstance(payload, dict) else payload
    samples: list[AmanahSample] = []; rejected = 0
    for ayah in ayahs:
        references = [ReferenceTranslation.model_validate(r) for r in ayah["references"]]
        faithful = build_faithful_samples(ayah_id=str(ayah["ayah_id"]),surah=int(ayah["surah"]),source_ar=str(ayah["source_ar"]),references=references)
        samples.extend(faithful)
        for base in faithful:
            for operator in OPERATORS:
                mutation = operator(base)
                if mutation is None: continue
                verdict = validate_mutation(base, mutation, expected_source_ar=ayah["source_ar"])
                if not verdict.valid: rejected += 1; continue
                samples.append(AmanahSample(sample_id=f"{base.sample_id}_{mutation.label.value.lower()}",ayah_id=base.ayah_id,surah=base.surah,source_ar=base.source_ar,references_en=base.references_en,candidate_en=mutation.candidate,labels=[mutation.label],severity=SEVERITY[mutation.label],origin="synthetic_mutation",mutation_type=mutation.label,changed_span=mutation.changed_span,metadata={**base.metadata,"mutation_operator": operator.__name__,"validation": "passed"}))
    splits = group_split(samples, 0.8, 0.1, 0.1, seed=seed); assert_no_ayah_leakage(splits)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items(): _dump_jsonl(out / f"{name}.jsonl", rows)
    summary = {"total_samples": len(samples),"faithful_samples": sum(DriftLabel.FAITHFUL in s.labels for s in samples),"synthetic_samples": sum(s.origin == "synthetic_mutation" for s in samples),"rejected_mutations": rejected,"ayahs": len({s.ayah_id for s in samples}),"split_sizes": {k: len(v) for k, v in splits.items()},"seed": seed}
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Build AMANAH-SD JSONL splits from an immutable reference bundle"); p.add_argument("source"); p.add_argument("output"); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); print(json.dumps(prepare_dataset(args.source, args.output, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
