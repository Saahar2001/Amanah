from __future__ import annotations

import argparse
import json

from scripts.fetch_verified_sources import APPROVED_ENGLISH_KEYS,QURANENC_LIST_URL,QURANENC_SURA_URL,TANZIL_DOWNLOAD_URL,TANZIL_FORM,parse_quranenc_sura,parse_tanzil_txt2,select_english_translations


def source_contract_smoke(*,session=None,timeout:int=60,expected_tanzil_ayahs:int=6236,expected_sura1_ayahs:int=7)->dict:
    if session is None:
        import requests
        session=requests.Session()
    tanzil=session.post(TANZIL_DOWNLOAD_URL,data=TANZIL_FORM,timeout=timeout); tanzil.raise_for_status(); arabic=parse_tanzil_txt2(tanzil.text)
    if len(arabic)!=expected_tanzil_ayahs: raise ValueError(f"Expected {expected_tanzil_ayahs} Tanzil ayahs, got {len(arabic)}")
    listing=session.get(QURANENC_LIST_URL,timeout=timeout); listing.raise_for_status(); metadata=select_english_translations(listing.json()); keys=[str(row['key']) for row in metadata]
    sura1_counts={}
    for key in APPROVED_ENGLISH_KEYS:
        response=session.get(QURANENC_SURA_URL.format(key=key,sura=1),timeout=timeout); response.raise_for_status(); parsed=parse_quranenc_sura(response.json())
        if len(parsed)!=expected_sura1_ayahs: raise ValueError(f"Expected {expected_sura1_ayahs} sura-1 ayahs for {key}, got {len(parsed)}")
        sura1_counts[key]=len(parsed)
    return {'status':'ok','tanzil_ayahs':len(arabic),'quranenc_translation_keys':keys,'quranenc_sura1_counts':sura1_counts}


def main()->None:
    p=argparse.ArgumentParser(description='Smoke-check AMANAH upstream source contracts'); p.add_argument('--timeout',type=int,default=60); args=p.parse_args(); print(json.dumps(source_contract_smoke(timeout=args.timeout),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
