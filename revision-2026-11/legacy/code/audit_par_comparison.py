"""Audit PAR artifacts against the frozen core; never publish incomplete results."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from summarize_revision import ROOT, aggregate, digest, read_runs


def audit(path):
    suite = json.loads(path.read_text())
    if suite['spec']['pilot']:
        raise ValueError('Pilot output is not scientific evidence')
    core = json.loads((ROOT / 'experiments/core-suite/suite.json').read_text())
    if core['status'] != 'complete':
        raise ValueError('Complete core reference suite required')
    combined = copy.deepcopy(core)
    combined['runs'] = {key: run for key, run in core['runs'].items()
                        if key.endswith(('_pretrain', '_retrain'))}
    expected = {f'{setting}_seed{seed}_par' for setting in ('iid', 'noniid')
                for seed in (42, 43, 44)}
    if set(suite['runs']) - expected:
        raise ValueError('Unexpected PAR run keys')
    markers = []
    for key, run in suite['runs'].items():
        if run['status'] != 'complete':
            continue
        folder = Path(run['run_dir'])
        protocol = json.loads((folder / 'par_protocol.json').read_text())
        config = json.loads((folder / 'config.json').read_text())
        metrics = json.loads((folder / 'metrics.json').read_text())
        if protocol['method'] != 'PAR-code-based adaptation':
            raise ValueError('Wrong method marker')
        if protocol['source_commit'] != suite['spec']['source_commit']:
            raise ValueError('PAR source commit mismatch')
        expected_fields = {'generator_steps': 40, 'generator_lr': .001,
                           'noise_l2': .01, 'repair_rounds': 10,
                           'local_epochs_per_stage': 1, 'local_lr': .001,
                           'input_size': 32,
                           'checkpoint_selection': 'final fixed round; no test selection',
                           'event_aggregation': 'equal floating states; max integer BN counters'}
        if any(protocol[field] != value for field, value in expected_fields.items()):
            raise ValueError('PAR event protocol changed')
        if config['model'] != 'resnet18' or config['data'] != 'cifar10' or config['num_clients'] != 10:
            raise ValueError('PAR shared-condition mismatch')
        if protocol['synthetic_count_per_target'] != 8 * config['batch_size']:
            raise ValueError('Wrong synthetic sample count')
        if not np.isfinite(protocol['objective_history']).all() or len(protocol['objective_history']) != 40:
            raise ValueError('Invalid generator history')
        if protocol['all_clients'] != list(range(10)):
            raise ValueError('Wrong PAR participation')
        hashes = json.loads((folder / 'par_source_hashes.json').read_text())
        for name, expected_hash in suite['spec']['hashes'].items():
            filename = 'PAR_COMPARISON_PROTOCOL.md' if name == 'protocol' else name
            if hashes[filename] != expected_hash or digest(folder / filename) != expected_hash:
                raise ValueError('PAR frozen source mismatch: ' + filename)
        progress = json.loads((folder / 'par_progress.json').read_text())
        if len(progress) != 11 or [p['repair_round'] for p in progress] != list(range(11)):
            raise ValueError('PAR requires impair plus ten repair aggregations')
        if [p['stage'] for p in progress] != ['impair'] + ['repair'] * 10:
            raise ValueError('Wrong PAR stage sequence')
        steps = [p['local_optimizer_steps'] for p in progress]
        if any(b <= a for a, b in zip([0] + steps, steps)) or steps[-1] != metrics['optimizer_steps']:
            raise ValueError('PAR optimizer accounting mismatch')
        prefix = key.rsplit('_', 1)[0]
        # Derive tensor size from the core one-stage event, whose upload is one model per client.
        event_folder = Path(core['runs'][prefix + '_noise']['run_dir'])
        event = json.loads((event_folder / 'metrics.json').read_text())['communication']
        participants = event['participating_clients']
        model_bytes = event['model_upload_bytes'] // len(participants)
        communication = metrics['communication']
        for name in ('model_download_bytes', 'model_upload_bytes'):
            if communication[name] != model_bytes * 10 * 11:
                raise ValueError('PAR tensor payload mismatch')
        if communication['noise_download_bytes'] != 8 * config['batch_size'] * 3 * 32 * 32 * 4 * 10:
            raise ValueError('PAR synthetic payload mismatch')
        combined['runs'][key] = run
        markers.append({'key': key, 'protocol_sha256': digest(folder / 'par_protocol.json'),
                        'progress_sha256': digest(folder / 'par_progress.json')})
    # Reuse the existing full accuracy, distance, attack split, class-label,
    # source, before-model and reference-hash audits on these matched runs.
    rows, evidence, missing = read_runs(combined)
    core_rows, _, core_missing = read_runs(core)
    if core_missing:
        raise ValueError('Incomplete core evidence')
    par_rows = [row for row in rows if row['method'] == 'par']
    comparisons = []
    for row in par_rows:
        for method in ('noise', 'unsir', 'ga', 'ft', 'retrain'):
            other = next(r for r in core_rows if r['setting'] == row['setting']
                         and r['seed'] == row['seed'] and r['method'] == method)
            fields = ('fa', 'ra', 'target_macro_auc', 'tv', 'seconds',
                      'local_optimizer_steps', 'event_payload_bytes')
            comparisons.append({'setting': row['setting'], 'seed': row['seed'],
                                'comparison': 'PAR minus ' + method,
                                **{field: row[field] - other[field] for field in fields
                                   if field in row and field in other}})
    complete = (suite['status'] == 'complete' and len(par_rows) == 6
                and set(suite['runs']) == expected and not missing)
    report = {'suite_sha256': digest(path), 'complete': complete,
              'completed_par_events': len(par_rows), 'planned_par_events': 6,
              'pending': sorted(expected - {m['key'] for m in markers}),
              'interpretation': 'PAR-code-based adaptation; native budgets; not an exact paper reproduction',
              'rows': par_rows, 'summary': aggregate(par_rows),
              'paired_differences': comparisons, 'markers': markers, 'evidence': evidence}
    output = path.parent / 'summary'
    output.mkdir(exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Audited {len(par_rows)}/6 PAR events; complete={complete}; {output}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='experiments/par-comparison/suite.json')
    args = parser.parse_args()
    audit(ROOT / args.suite)
