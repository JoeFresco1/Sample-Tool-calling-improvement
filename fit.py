"""Closed-form, constrained output-row scale correction from observed activations."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from support import P,ROOT,api
sys.path.insert(0,str(ROOT/'tuning/attention-20261031/source'))
from tools.artifact.reader import Artifact
from tools.artifact.layouts import row_split_geometry


def bf16(path):
    return (np.fromfile(path,dtype='<u2').astype(np.uint32)<<16).view('<f4').astype(np.float64)


def main():
    data=json.loads((P/'data-frozen.json').read_text())
    contexts=data['contexts']
    observed={r['id']:r for r in json.loads((P/'observer-contexts.json').read_text())}
    assert len(observed)==len(contexts)
    mismatch=[c['id'] for c in contexts if observed[c['id']]['native_text']!=observed[c['id']]['observer_text']]
    # First-token references must agree before using a diagnostic-built model's activations.
    if mismatch:raise RuntimeError(f'Observer/native decisions differ: {mismatch}')
    x=np.stack([bf16(P/'activations'/(c['id']+'-x.bf16')) for c in contexts])
    logits=np.stack([bf16(P/'activations'/(c['id']+'-logits.bf16')) for c in contexts])
    top=np.argmax(logits,axis=1)
    text={}
    def decode(token):
        token=int(token)
        if token not in text:text[token]=api('detokenize',{'tokens':[token]},port=8087)['content']
        return text[token]
    edits=[];selected=set()
    for i,c in enumerate(contexts):
        is_edit=c['kind']!='calibration'
        actual=decode(top[i])
        valid=(actual.lstrip().startswith('[') if c['kind']=='array'
               else actual.lstrip() in ('tool','terminal')) if is_edit else True
        if is_edit and not valid:
            wanted=int(c['target']);wrong=int(top[i]);gap=float(logits[i,wrong]-logits[i,wanted])
            edits.append({'index':i,'id':c['id'],'wanted':wanted,'wrong':wrong,
                          'wrong_text':actual,'gap':gap})
            selected.update((wanted,wrong))
    if not edits:raise RuntimeError('No wrong captured decisions; no justified patch target')
    if len(selected)>10:raise RuntimeError('Target set exceeds fixed small-edit scope')
    source=ROOT/'models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer'
    with Artifact(source) as artifact,source.open('rb') as f:
        obj=artifact.object(artifact.directory.bindings['text/output_head']['object'])
        assert obj.shape==(248320,5120) and obj.format=='t2_g128_fp16'
        geo=row_split_geometry(obj.format,obj.shape)
        assert geo.groups_per_row==40 and geo.base_row_bytes==1280 and geo.scale_row_bytes==80
        rows=[];oracle=[]
        for token in sorted(selected):
            code_offset=artifact.payload_offset+obj.offset+token*1280
            scale_offset=artifact.payload_offset+obj.offset+geo.scale_offset+token*80
            f.seek(code_offset);packed=f.read(1280)
            f.seek(scale_offset);raw_scales=f.read(80)
            codes=np.frombuffer(packed,dtype=np.uint8)
            unpacked=((codes[:,None]>>np.array([0,2,4,6],dtype=np.uint8))&3).reshape(-1).astype(np.int16)
            assert not np.any(unpacked==2)
            signed=np.where(unpacked==3,-1,unpacked).reshape(40,128).astype(np.float64)
            # Independent scalar oracle on 128 channels, separate from vectorized bit unpacking.
            for k in range(128):
                q=(packed[k//4]>>(2*(k%4)))&3
                assert signed.flat[k]==(q-4 if q>=2 else q)
            scales=np.frombuffer(raw_scales,dtype='<f2').astype(np.float64)
            assert np.all(np.isfinite(scales)) and np.all(scales>0)
            features=np.einsum('sgk,gk->sg',x.reshape(-1,40,128),signed)*scales
            reconstructed=features.sum(axis=1)
            actual=logits[:,token]
            error=np.abs(reconstructed-actual)
            max_error=float(error.max())
            # BF16 output rounding plus a small private accumulation allowance.
            allowance=np.maximum(.04,np.abs(actual)*.006)
            if np.any(error>allowance):
                raise RuntimeError(f'Selected-row reconstruction fails: token {token}, max error {max_error}, worst ratio {float((error/allowance).max())}')
            oracle.append({'token':token,'text':decode(token),'max_abs_error':max_error,
                           'max_error_over_allowance':float((error/allowance).max())})
            y=np.zeros(len(contexts));weights=np.full(len(contexts),4.0)
            for e in edits:
                if token==e['wanted']:y[e['index']]=max(1.5,e['gap']+1.5);weights[e['index']]=3.0
                elif token==e['wrong']:y[e['index']]=-.5*max(1.5,e['gap']+1.5);weights[e['index']]=3.0
            a=features*np.sqrt(weights[:,None]);b=y*np.sqrt(weights)
            gram=a.T@a
            ridge=.05*float(np.trace(gram))/40
            relative=np.linalg.solve(gram+np.eye(40)*ridge,a.T@b)
            relative=np.clip(relative,-.5,.5)
            rows.append({'token_id':token,'text':decode(token),'scale_offset':scale_offset,
                'original_hex':raw_scales.hex(),'relative_delta':relative.tolist(),
                'max_abs_relative_delta':float(np.max(np.abs(relative))),
                'target_changes':y.tolist(),'predicted_changes':(features@relative).tolist()})
    result={'method':'regularized closed-form selected output-row group-scale correction',
            'data_sha256':hashlib.sha256((P/'data-frozen.json').read_bytes()).hexdigest(),
            'source':str(source),'head_object':obj.id,'head_shape':list(obj.shape),
            'strengths':[.25,.5,1.0],'relative_scale_change_bound':.5,'ridge_trace_fraction':.05,
            'edit_contexts':edits,'observer_native_mismatches':mismatch,'oracle':oracle,'rows':rows,
            'scale_values_at_most':len(rows)*40,'affected_weight_entries_at_most':len(rows)*5120,
            'numpy_version':np.__version__}
    out=P/'fit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'scales':len(rows)*40,'edits':edits,'oracle':oracle}),flush=True)


if __name__=='__main__':main()
