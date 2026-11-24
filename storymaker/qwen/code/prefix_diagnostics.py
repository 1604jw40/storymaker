"""Post-hoc completed text-field diagnostics, including truncated JSON prefixes."""
import json
import re
import statistics
import unicodedata
from collections import Counter

from storymaker import project as p
def complete_text_fields(raw):
    decoder = json.JSONDecoder()
    result = []
    i = 0
    while i < len(raw):
        if raw[i] != '"':
            i += 1
            continue
        try:
            value, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        i = end
        while i < len(raw) and raw[i].isspace():
            i += 1
        if value != 'text' or i >= len(raw) or raw[i] != ':':
            continue
        i += 1
        while i < len(raw) and raw[i].isspace():
            i += 1
        try:
            value, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        if isinstance(value, str):
            result.append(value)
        i = end
    return result


def repeated(texts):
    normalized = [''.join(c for c in unicodedata.normalize('NFKC', s) if c.isalnum()) for s in texts]
    counts = Counter(s for s in normalized if len(s) >= 20)
    extra = sum(n - 1 for n in counts.values())
    return {'complete_text_fields': len(texts), 'extra_exact_repeated_texts': extra,
            'max_exact_multiplicity': max(counts.values(), default=0),
            'has_exact_repeat': extra > 0}


def main():
    run = p.read(p.ROOT.parent / 'outputs/sft_v5/active_run.json')['run_dir']
    from pathlib import Path
    rows = []
    for path in sorted((Path(run) / 'evaluation/generations/test').glob('*/*.json')):
        if path.stem not in ('base', 'sft'):
            continue
        generated = p.read(path)['generation']
        texts = complete_text_fields(generated['raw_text'])
        rows.append({'variant': path.stem, 'source_id': int(path.parent.name),
                     'eos': generated['ended_with_eos'], 'generated_tokens': generated['generated_tokens'],
                     **repeated(texts)})
    assert len(rows) == 122 and sum(not r['eos'] for r in rows if r['variant'] == 'sft') == 22
    groups = {}
    for variant in ('base', 'sft'):
        for eos in (True, False):
            group = [r for r in rows if r['variant'] == variant and r['eos'] == eos]
            groups[f'{variant}/eos={eos}'] = {'n': len(group),
                'with_any_complete_text': sum(r['complete_text_fields'] > 0 for r in group),
                'with_exact_repeat': sum(r['has_exact_repeat'] for r in group),
                'median_complete_text_fields': statistics.median(r['complete_text_fields'] for r in group),
                'extra_repeated_texts': sum(r['extra_exact_repeated_texts'] for r in group),
                'max_exact_multiplicity': max(r['max_exact_multiplicity'] for r in group)}
    report = {'scope': 'Post-hoc text-field scan of all122 preserved outputs, not new generations',
              'method': 'Scan complete JSON string values with key text; stop at an incomplete quoted string. Exact NFKC alphanumeric repeat, >=20characters.',
              'limitations': 'Partial text fields/endings are not recovered. Fields are not a proof of valid scenes. Refrains may be appropriate. Association with truncation is not causal evidence.',
              'groups': groups, 'rows': rows, 'new_generations': 0, 'new_api_calls': 0}
    p.save(p.ROOT / 'reports/posthoc_all_outputs_prefix_diagnostics.json', report)
    print(json.dumps(groups))


if __name__ == '__main__':
    main()
