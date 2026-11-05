"""Use actual generated tool-name decision states, with nested-name preservation."""
from pathlib import Path
import importlib.util,json,os,shutil,sys
import numpy as np

P=Path(__file__).resolve().parent;R2=P.parent
sys.path.insert(0,str(R2))
import fit as base_fit
from support import api


def main():
    data=json.loads((P/'data-frozen.json').read_text())
    trajectories={r['id']:r for r in json.loads((P/'trajectories.json').read_text())}
    activation=P/'activations';activation.mkdir(exist_ok=True)
    contexts=[];observed=[];selection=[];excluded=[]
    def link(old,new):
        if new.exists():assert os.path.samefile(old,new)
        else:os.link(old,new)
    for c in data['contexts_to_capture']:
        response=trajectories[c['id']]['response'];message=response['choices'][0]['message']
        calls=message.get('tool_calls') or []
        if not calls or calls[0]['function']['name']!=c['expected_name'] or (message.get('content') or '').strip():
            excluded.append({'id':c['id'],'reason':'Expected function/no-prose qualification failed',
                             'calls':[x['function']['name'] for x in calls],'content':message.get('content')});continue
        native=api('v1/chat/completions',c['body'],port=8087)
        native_calls=native['choices'][0]['message'].get('tool_calls') or []
        if not native_calls or native_calls[0]['function']['name']!=calls[0]['function']['name']:
            excluded.append({'id':c['id'],'reason':'Native/observer first function differs','native':native});continue
        snapshots=sorted((P/'captures').glob(c['id']+'-*-logits.bf16'))
        chosen=None
        token_id=None
        for f in snapshots:
            top=int(np.argmax(base_fit.bf16(f)))
            if top==c['state_token']:chosen=f;token_id=top;break
        if chosen is None and 'read' in c['id']:
            # The real XML renderer can merge '=read' into one token, unlike the JSON probe.
            for f in snapshots[:16]:
                top=int(np.argmax(base_fit.bf16(f)))
                fragment=api('detokenize',{'tokens':[top]},port=8087)['content']
                if 'read' in fragment:chosen=f;token_id=top;break
        if chosen is None:
            excluded.append({'id':c['id'],'reason':'No matching greedy token state'});continue
        stem=chosen.name.removesuffix('-logits.bf16')
        link(chosen,activation/(c['id']+'-logits.bf16'))
        link(P/'captures'/(stem+'-x.bf16'),activation/(c['id']+'-x.bf16'))
        contexts.append({'id':c['id'],'kind':c['kind'],'target':c['target']})
        observed.append({'id':c['id'],'native_text':native_calls[0]['function']['name'],
                         'observer_text':calls[0]['function']['name'],
                         'qualification':'First returned function names agree; actual observer token state selected separately. No raw native-logit equivalence claim.',
                         'native_response':native})
        selection.append({'id':c['id'],'snapshot':stem,'kind':c['kind'],'returned_function':calls[0]['function']['name'],'selected_raw_token':token_id})
    assert sum(c['kind']=='dispatch' for c in contexts)>=4
    assert sum('nested' in c['id'] for c in contexts)>=2
    assert sum('read' in c['id'] for c in contexts)>=2
    old_data=json.loads((R2/'data-frozen.json').read_text())
    old_observed={r['id']:r for r in json.loads((R2/'observer-contexts.json').read_text())}
    for c in old_data['contexts']:
        if c['kind']!='calibration':continue
        name='neutral-'+c['id']
        for suffix in ('-x.bf16','-logits.bf16'):link(R2/'activations'/(c['id']+suffix),activation/(name+suffix))
        contexts.append({'id':name,'kind':'calibration','target':c.get('target')})
        row=dict(old_observed[c['id']]);row['id']=name;observed.append(row)
    shutil.copyfile(P/'data-frozen.json',P/'data-request-plan.json')
    data['contexts']=contexts
    (P/'data-frozen.json').write_text(json.dumps(data,sort_keys=True),encoding='utf-8')
    (P/'observer-contexts.json').write_text(json.dumps(observed,indent=2),encoding='utf-8')
    (P/'state-selection.json').write_text(json.dumps({'selected':selection,'excluded':excluded,'neutral_count':20},indent=2),encoding='utf-8')
    base_fit.P=P
    base_fit.main()
    f=json.loads((P/'fit.json').read_text())
    f.update(strengths=[.5,1.,1.5],maximum_effective_relative_change=.75,
             state_source='Actual autoregressive function-name steps; nested write names and real read calls protected',
             observer_native_comparison='First returned function names for actual trajectories; first-token text for previous neutral contexts. Not full raw native-logit equivalence.')
    (P/'fit.json').write_text(json.dumps(f,indent=2),encoding='utf-8')
    print(json.dumps({'selected_actual_states':len(selection),'excluded':excluded,
                      'edit_contexts':len(f['edit_contexts']),'rows':len(f['rows'])}),flush=True)


if __name__=='__main__':main()
