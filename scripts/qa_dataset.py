from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data.models import TRAINABLE_DRIFT_LABELS_V0


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists(): raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def qa_dataset(reference_bundle: str | Path, split_dir: str | Path, *, expected_ayahs: int = 6236, expected_references: int = 3, require_trainable_labels: bool = True) -> dict:
    bundle = json.loads(Path(reference_bundle).read_text(encoding='utf-8')); ayahs = bundle.get('ayahs', bundle) if isinstance(bundle, dict) else bundle
    by_ayah: dict[str, dict] = {}; duplicate_ayah_ids=[]; bad_reference_counts=[]; empty_canonical=[]
    for row in ayahs:
        aid=str(row.get('ayah_id',''))
        if aid in by_ayah: duplicate_ayah_ids.append(aid)
        by_ayah[aid]=row
        if not str(row.get('source_ar','')).strip(): empty_canonical.append(aid)
        if len(row.get('references',[])) != expected_references: bad_reference_counts.append(aid)
    split_path=Path(split_dir); splits={name:_read_jsonl(split_path/f'{name}.jsonl') for name in ('train','validation','test')}; split_sets={name:{str(r['ayah_id']) for r in rows} for name,rows in splits.items()}
    leakage=sorted((split_sets['train']&split_sets['validation'])|(split_sets['train']&split_sets['test'])|(split_sets['validation']&split_sets['test']))
    canonical_mismatches=[]; unknown_ayahs=[]; label_counts=Counter(); origin_counts=Counter()
    for rows in splits.values():
        for sample in rows:
            aid=str(sample['ayah_id']); ref=by_ayah.get(aid)
            if ref is None: unknown_ayahs.append(aid)
            elif str(sample.get('source_ar','')) != str(ref.get('source_ar','')): canonical_mismatches.append(aid)
            label_counts.update(str(x) for x in sample.get('labels',[])); origin_counts.update([str(sample.get('origin','unknown'))])
    required_labels=[x.value for x in TRAINABLE_DRIFT_LABELS_V0]; missing_trainable_labels=[x for x in required_labels if label_counts[x]==0]
    failures=[]
    if len(by_ayah)!=expected_ayahs: failures.append(f'expected {expected_ayahs} reference ayahs, got {len(by_ayah)}')
    if duplicate_ayah_ids: failures.append('duplicate ayah ids')
    if bad_reference_counts: failures.append('unexpected reference count')
    if empty_canonical: failures.append('empty canonical Arabic')
    if leakage: failures.append('ayah leakage across splits')
    if canonical_mismatches: failures.append('canonical Arabic mismatch')
    if unknown_ayahs: failures.append('samples with unknown ayah ids')
    if require_trainable_labels and missing_trainable_labels: failures.append('missing trainable labels')
    return {'status':'ok' if not failures else 'failed','failures':failures,'reference_ayahs':len(by_ayah),'expected_ayahs':expected_ayahs,'expected_references_per_ayah':expected_references,'duplicate_ayah_ids':sorted(set(duplicate_ayah_ids)),'bad_reference_count_ayahs':sorted(set(bad_reference_counts)),'empty_canonical_ayahs':sorted(set(empty_canonical)),'split_sizes':{k:len(v) for k,v in splits.items()},'split_ayah_counts':{k:len(v) for k,v in split_sets.items()},'split_ayah_leakage':leakage,'canonical_mismatches':sorted(set(canonical_mismatches)),'unknown_ayahs':sorted(set(unknown_ayahs)),'label_counts':dict(sorted(label_counts.items())),'origin_counts':dict(sorted(origin_counts.items())),'missing_trainable_labels':missing_trainable_labels}


def main() -> None:
    p=argparse.ArgumentParser(description='QA AMANAH reference bundle and generated split dataset'); p.add_argument('--reference-bundle',required=True); p.add_argument('--split-dir',required=True); p.add_argument('--output',default='artifacts/amanah_sd/qa_report.json'); p.add_argument('--expected-ayahs',type=int,default=6236); p.add_argument('--expected-references',type=int,default=3); p.add_argument('--allow-missing-labels',action='store_true'); args=p.parse_args()
    report=qa_dataset(args.reference_bundle,args.split_dir,expected_ayahs=args.expected_ayahs,expected_references=args.expected_references,require_trainable_labels=not args.allow_missing_labels); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['status']!='ok': raise SystemExit(2)


if __name__=='__main__': main()
