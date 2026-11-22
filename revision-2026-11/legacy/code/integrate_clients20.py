"""Audited same-seed client-count comparison; text only."""
import json
from summarize_revision import ROOT,read_runs,aggregate,stats
from summarize_followup import table,pm

def main():
    suites=[json.loads((ROOT/'experiments'/n/'suite.json').read_text()) for n in ('core-suite','extension-clients20')]
    allrows=[]
    for suite in suites:
        rows,_,missing=read_runs(suite)
        if missing or suite['status']!='complete':raise ValueError('Incomplete evidence')
        allrows.append(rows)
    core,extra=allrows; groups=aggregate(extra); paired={}
    for r in extra:
        original=next(x for x in core if x['seed']==r['seed'] and x['setting']==r['setting'] and x['method']==r['method'])
        g=paired.setdefault(r['setting']+'/'+r['method'],{})
        for m in ('fa','ra','target_macro_auc','seconds','event_payload_bytes','local_optimizer_steps'):
            if m in r and m in original:g.setdefault(m,[]).append(r[m]-original[m])
    paired={g:{m:stats(v) for m,v in ms.items()} for g,ms in paired.items()}
    lines=[r'\subsection{Twenty-Client Scale Extension}',r'\label{sec:clients20}',
        'The twenty-client extension has 36 complete audited stages, IID and '
        '$\\alpha=0.5$, seeds 42--44. The same CIFAR-10 dataset and ResNet18 '
        'recipe retain 50 rounds, five local epochs and 40\\% participation; '
        'eight clients participate per round rather than four. Smaller local '
        'partitions, minibatch rounding and client schedules change jointly. '
        'This is a scale extension, not an isolated causal or equal-time test. '
        'A separate twelve-stage pilot passed before formal training. Mean '
        '$\\pm$ sample SD uses $n=3$.']
    for setting in ('iid','noniid'):
        lines.append(table('Twenty-client CIFAR-10/ResNet18 '+setting+' results, three seeds.',
            'tab:clients20-'+setting,groups,[(setting+'/'+m,n) for m,n in [('pretrain','Original'),('retrain','Retrain'),('ft','Finetune'),('ga','GA'),('noise','Fed-EMNU'),('unsir','UNSIR-style')]]))
        g=paired[setting+'/noise'];n=groups[setting+'/noise']
        lines.append(('IID' if setting=='iid' else 'Non-IID')+' paired twenty-minus-ten client '
            'Fed-EMNU RA difference is '+pm(g,'ra',100)+' percentage points, FA difference '+
            pm(g,'fa',100)+' points, and target AUC difference '+pm(g,'target_macro_auc',digits=3)+
            '. These differences pair seeds 42--44 only; the five-seed primary '
            'population is not mixed into this comparison. Twenty-client event '
            'payload averages '+f"{n['event_payload_bytes']['mean']/2**20:.2f}"+
            ' MiB, with '+pm(n,'local_optimizer_steps')+' local optimizer steps. '
            'Native compute and payload changes are reported rather than interpreted '
            'as real-network scalability guarantees.')
    lines.append('Two simulated client counts on one GPU do not establish distributed '
        'network performance, device heterogeneity or monotonic scalability. All '
        'methods share their twenty-client starting checkpoint and reference within '
        'seed. The original target-holder-only event and its sparse-ownership limitation '
        'are unchanged. No twenty-client PAR result is claimed.')
    (ROOT/'manuscript/revision_clients20.tex').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'experiments/extension-clients20/paired10-summary.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    p=ROOT/'manuscript/Fed-EMNU_MDPI.tex';t=p.read_text(encoding='utf-8')
    if '\\input{revision_clients20.tex}' not in t:t=t.replace('\\input{revision_sequence.tex}','\\input{revision_sequence.tex}\n\\input{revision_clients20.tex}')
    t=t.replace('one ResNet family, ten clients and a fixed two-request','one ResNet family, ten/twenty simulated clients and a fixed two-request')
    t=t.replace('Larger client counts and\npost-event federated training remain untested.',
        'The twenty-client extension in Section~\\ref{sec:clients20} is a limited scale test;\npost-event federated training remains under investigation.')
    p.write_text(t,encoding='utf-8')
    p=ROOT/'responses/EDITOR_REQUIREMENT_MATRIX.md';t=p.read_text(encoding='utf-8').replace('| Larger client numbers (R1 recommendation) | Not evaluated | All runs have ten clients; scalability unestablished |',
        '| Larger client numbers (R1 recommendation) | Addressed for20-client simulation | 36 audited stages, IID/NonIID,n3, paired10-versus20 comparison; real-network/large-scale scalability unestablished |')
    p.write_text(t,encoding='utf-8')
    p=ROOT/'responses/response-to-reviewers.md';t=p.read_text(encoding='utf-8')
    t+='\n\n## Twenty-client scale follow-up\n\nThe 36-stage twenty-client experiment is complete and audited, with12 separate pilot stages excluded. Section5.10 reports utility, forgetting, class-conditioned attacks and same-seed42–44 differences from ten clients. Participation remains40% (eight clients per round); partitions/minibatch counts change jointly. Actual transport is unmeasured and two counts do not establish general scalability. This supersedes earlier statements that larger client counts were not evaluated. Post-event FL continuation remains a separate pending check.\n'
    p.write_text(t,encoding='utf-8')
    print('Twenty-client tables/paired comparison integrated; no PDF generated')
if __name__=='__main__':main()
