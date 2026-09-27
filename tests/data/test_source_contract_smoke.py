class FakeResponse:
    def __init__(self, *, text='', payload=None):
        self.text = text; self._payload = payload
    def raise_for_status(self): return None
    def json(self): return self._payload


class FakeSession:
    def __init__(self): self.calls = []
    def post(self, url, data=None, timeout=None):
        self.calls.append(('POST', url, data, timeout)); return FakeResponse(text='1|1|ا\n1|2|ب\n')
    def get(self, url, timeout=None):
        self.calls.append(('GET', url, None, timeout))
        if 'translations/list' in url:
            return FakeResponse(payload=[{'key':'english_rwwad','version':'1'},{'key':'english_saheeh','version':'1'},{'key':'english_hilali_khan','version':'1'}])
        return FakeResponse(payload=[{'sura':'1','aya':'1','translation':'A'}])


def test_source_contract_smoke_checks_tanzil_and_all_three_quranenc_refs_without_full_download():
    from scripts.source_contract_smoke import source_contract_smoke
    session=FakeSession(); report=source_contract_smoke(session=session,expected_tanzil_ayahs=2,expected_sura1_ayahs=1)
    assert report['status']=='ok'; assert report['tanzil_ayahs']==2; assert report['quranenc_translation_keys']==['english_rwwad','english_saheeh','english_hilali_khan']
    assert len([c for c in session.calls if '/translation/sura/' in c[1]])==3
