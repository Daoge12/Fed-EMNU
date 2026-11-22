"""Generate separate follow-up tables from complete, audited CIFAR-10 suites."""
import json
from pathlib import Path
from summarize_revision import ROOT, read_runs, aggregate, paired, digest
from audit_par_comparison import audit


def pm(group, key, scale=1, digits=2):
    value = group[key]
    return f"${value['mean']*scale:.{digits}f} \\pm {value['sd']*scale:.{digits}f}$"


def table(caption, label, groups, methods):
    lines = [r'\begin{table}[H]', '\\caption{' + caption + '}',
             '\\label{' + label + '}', r'\centering\scriptsize\setlength{\tabcolsep}{3.5pt}',
             r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Method & FA (\%) & RA (\%) & Target AUC & TV & Time (s)\\', r'\midrule']
    for key, name in methods:
        g = groups[key]
        lines.append(name + ' & ' + ' & '.join([pm(g, 'fa', 100), pm(g, 'ra', 100),
                     pm(g, 'target_macro_auc', digits=3), pm(g, 'tv', digits=3)
                     if 'tv' in g else '---', pm(g, 'seconds')]) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    reports = {}
    for name, planned in (('extension-resnet34', 36), ('extension-alpha01', 30)):
        path = ROOT / 'experiments' / name / 'suite.json'
        suite = json.loads(path.read_text())
        rows, evidence, missing = read_runs(suite)
        if suite['status'] != 'complete' or missing or len(rows) != planned:
            raise ValueError('Complete follow-up required: ' + name)
        reports[name] = {'suite_sha256': digest(path), 'summary': aggregate(rows),
                         'paired_selection': paired(rows), 'evidence': evidence}
    par = audit(ROOT / 'experiments/par-comparison/suite.json')
    if not par['complete']:
        raise ValueError('Complete PAR comparison required')
    core = json.loads((ROOT / 'experiments/core-suite/summary/summary.json').read_text())['summary']
    lines = [r'\subsection{Completed Follow-Up Experiments}', r'\label{sec:followup}',
        'The follow-up retains the fixed training recipe and three seeds (42--44). '
        'It adds 36 CIFAR-10/ResNet34 stages (six methods in each partition), '
        '30 ResNet18 stages at Dirichlet $\\alpha=0.1$ (including controls), and '
        'six PAR-code-based events on the original core checkpoints. '
        'These extensions do not replace or tune the core results. All completed '
        'artifacts pass the same accuracy, reference-distance, source and attack '
        'accounting audits; PAR additionally checks its fixed eleven aggregation '
        'stages and method-specific provenance. Values below are mean $\\pm$ '
        'sample SD, $n=3$. No additional dataset or client count is evaluated.']
    names = [('pretrain', 'Original'), ('retrain', 'Retrain'), ('ft', 'Finetune'),
             ('ga', 'GA'), ('noise', 'Fed-EMNU'), ('unsir', 'UNSIR-style')]
    g34 = reports['extension-resnet34']['summary']
    for setting, title in (('iid', 'IID'), ('noniid', 'Non-IID, $\\alpha=0.5$')):
        lines.append(table('ResNet34 architecture extension on CIFAR-10 (' + title +
                           '). TV is against the corresponding ResNet34 retained-data reference.',
                           'tab:r34-' + setting, g34, [(setting+'/'+m, n) for m,n in names]))
    lines.append('This within-family architecture extension does not establish architecture '
                 'independence. The same fixed recipe is evaluated without test-based retuning; '
                 'its utility, forgetting and attack values must be considered jointly.')
    ga = reports['extension-alpha01']['summary']
    lines.append(table('Stronger heterogeneity on CIFAR-10/ResNet18 ($\\alpha=0.1$). '
                       'Low initial utility limits the interpretation of low final FA.',
                       'tab:alpha01', ga, [('noniid/'+m, n) for m,n in names]))
    original, retrain, noise = [ga['noniid/'+m] for m in ('pretrain','retrain','noise')]
    selection = reports['extension-alpha01']['paired_selection']['noniid']['confidence_minus_random']
    participants = {}
    suite = json.loads((ROOT / 'experiments/extension-alpha01/suite.json').read_text())
    for seed in (42,43,44):
        folder = Path(suite['runs'][f'noniid_seed{seed}_noise']['run_dir'])
        participants[seed] = json.loads((folder/'metrics.json').read_text())['communication']['participating_clients']
    lines += ['At $\\alpha=0.1$, Original RA is '+pm(original,'ra',100)+
              '\\%, Retrain RA is '+pm(retrain,'ra',100)+'\\%, and Fed-EMNU RA is '+
              pm(noise,'ra',100)+'\\%. Poor retained utility is therefore present before '
              'unlearning. Fed-EMNU target AUC is '+pm(noise,'target_macro_auc',digits=3)+
              ', versus '+pm(retrain,'target_macro_auc',digits=3)+' for Retrain. '
              'Confidence minus random RA at $rt=0.5$ is '+pm(selection,'ra',100)+
              ' percentage points and TV changes by '+pm(selection,'tv',digits=3)+
              '. These controls again provide no support for a confidence-ranking advantage.',
              'The Fed-EMNU event includes '+', '.join(str(len(participants[s])) for s in (42,43,44))+
              ' target-holding clients for seeds 42, 43 and 44, respectively. '
              'Replacing the global state with updates from only those clients can omit '
              'unaffected-client contributions. This is a protocol limitation under sparse '
              'target ownership; it is not evidence of robust heterogeneity handling.']
    groups = {**core, **par['summary']}
    for setting, title in (('iid', 'IID'), ('noniid', 'Non-IID, $\\alpha=0.5$')):
        lines.append(table('PAR-code-based adaptation on shared core ResNet18 checkpoints ('+
                           title+'). Native method budgets differ; this is not an equal-compute ablation.',
                           'tab:par-'+setting, groups, [(setting+'/'+m,n) for m,n in
                           [('retrain','Retrain'),('noise','Fed-EMNU'),('unsir','UNSIR-style'),('par','PAR adaptation')]]))
    lines += ['Under this code-based adaptation, PAR retains substantial target-class '
              'accuracy and its target MIA AUC remains above Retrain in both partitions. '
              'The ten repair rounds can restore utility and target behavior together; '
              'the experiment does not isolate that mechanism causally. These outcomes '
              'describe the specified shared-checkpoint adaptation, not a reproduction '
              'or refutation of the published PAR results.',
              'PAR uses about $9.974$ GB of model and synthetic tensor payload per '
              'event (decimal units; transport framing excluded). It communicates one '
              'model download and upload per client at each of eleven aggregations. '
              'Its native multi-round budget, generator, learning rates, equal event '
              'aggregation and all-client participation differ jointly from Fed-EMNU. '
              'The separate audit records paired differences in time, local optimizer '
              'steps and payload; observed differences cannot be assigned to one component.']
    output = ROOT / 'manuscript/revision_followup.tex'
    output.write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    (ROOT/'experiments/followup-summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
