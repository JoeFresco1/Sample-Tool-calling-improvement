"""Modify only explicitly fitted output-head scale words in a physical model copy."""
from pathlib import Path
import hashlib,json,shutil,struct,uuid
P=Path(__file__).resolve().parent

MODEL=P/'head-edited.ninfer'


def patch(strength):
    fit=json.loads((P/'fit.json').read_text())
    assert strength in fit['strengths']
    source=Path(fit['source'])
    if not MODEL.exists():shutil.copyfile(source,MODEL)
    assert MODEL.stat().st_size==source.stat().st_size and MODEL.resolve()!=source.resolve()
    changes=[];new_id=uuid.uuid4().bytes
    with MODEL.open('r+b') as f:
        f.seek(16);f.write(new_id)
        for row in fit['rows']:
            before=bytes.fromhex(row['original_hex'])
            old=[v[0] for v in struct.iter_unpack('<e',before)]
            after=b''.join(struct.pack('<e',v*(1+strength*d)) for v,d in zip(old,row['relative_delta']))
            values=[v[0] for v in struct.iter_unpack('<e',after)]
            assert all(0<v<65504 for v in values)
            assert all(.249<=b/a<=1.751 for a,b in zip(old,values))
            f.seek(row['scale_offset']);f.write(after)
            changes.append({'token_id':row['token_id'],'text':row['text'],'file_offset':row['scale_offset'],
                'before_hex':before.hex(),'after_hex':after.hex(),
                'changed_scales':sum(a!=b for a,b in zip(old,values))})
        f.flush()
    manifest={'strength':strength,'model':str(MODEL),'artifact_id':new_id.hex(),'changes':changes,
              'ternary_codes_changed':False,'format_changed':False,'bytes':MODEL.stat().st_size}
    (P/f'patch-{strength:g}.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def verify(strength):
    fit=json.loads((P/'fit.json').read_text());source=Path(fit['source'])
    manifest=json.loads((P/f'patch-{strength:g}.json').read_text())
    ranges=[(16,32)]+[(r['file_offset'],r['file_offset']+80) for r in manifest['changes']]
    ranges.sort();unchanged=0;original_sha=hashlib.sha256();edited_sha=hashlib.sha256()
    with source.open('rb') as a,MODEL.open('rb') as b:
        cursor=0
        for lo,hi in ranges+[(source.stat().st_size,source.stat().st_size)]:
            remaining=lo-cursor;assert remaining>=0
            while remaining:
                count=min(8*1024*1024,remaining);x,y=a.read(count),b.read(count)
                assert x==y and len(x)==count
                original_sha.update(x);edited_sha.update(y);unchanged+=count;remaining-=count;cursor+=count
            x,y=a.read(hi-lo),b.read(hi-lo);original_sha.update(x);edited_sha.update(y);cursor=hi
        assert not a.read(1) and not b.read(1)
    for row in manifest['changes']:
        with MODEL.open('rb') as f:f.seek(row['file_offset']);assert f.read(80).hex()==row['after_hex']
    result={'source_sha256':original_sha.hexdigest(),'edited_sha256':edited_sha.hexdigest(),
            'bytes_identical_outside_declared_scale_words_and_id':unchanged,'bytes':MODEL.stat().st_size,
            'changed_scales':sum(r['changed_scales'] for r in manifest['changes']),
            'strength':strength,'ternary_codes_unchanged':True}
    (P/'patch-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
