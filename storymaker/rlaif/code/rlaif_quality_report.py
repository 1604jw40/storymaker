"""Failure-aware paired RL comparisons, CE, prefix repetition and literal-copy candidates."""
import argparse
import csv
import difflib
import json
import math
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from storymaker import project as p
from storymaker.qwen.code.evaluation_api import CRITERIA
from storymaker.qwen.code.evaluation_report import interval
from storymaker.qwen.code.prefix_diagnostics import complete_text_fields, repeated
from storymaker.rlaif.code.rlaif_environment import OUT
from storymaker.rlaif.code.rlaif_evaluate import local_metrics, sha
from storymaker.rlaif.code.rlaif_quality_api import validate, source_pairs, verify_generated_sources
from storymaker.rlaif.code.rlaif_sequence import save
from storymaker.rlaif.code.rlaif_training_report import main as write_training_report, committed_batch, committed_batches, run_lineage


def copy_candidates(references, outputs, minimum=40):
    def normalized(raw):
        return ''.join(c for c in unicodedata.normalize('NFKC',' '.join(complete_text_fields(raw))) if c.isalnum())
    texts = {row['id']: normalized(row['raw_text']) for row in references}
    width = minimum//2
    index = defaultdict(set)
    for key,text in texts.items():
        for position in range(0,len(text)-width+1,width):
            index[text[position:position+width]].add(key)
    matches, checked, unavailable = [],0,[]
    for (number,variant),story in outputs.items():
        text = normalized(story['raw_text'])
        if not text:
            unavailable.append({'id':number,'variant':variant})
            continue
        checked += 1
        candidates = set()
        for position in range(len(text)-width+1):
            candidates.update(index.get(text[position:position+width],()))
        for key in sorted(candidates):
            found = difflib.SequenceMatcher(None,text,texts[key],autojunk=False).find_longest_match()
            if found.size >= minimum:
                matches.append({'id':number,'variant':variant,'reference_id':key,
                    'normalized_characters':found.size,'matched_text':text[found.a:found.a+found.size]})
    return {'checked_outputs':checked,'outputs_without_complete_text_field':unavailable,
        'minimum_normalized_characters':minimum,'reference_count':len(references),'matches':matches,
        'scope':'Complete text string fields only, including recoverable truncated prefixes; NFKC letters/digits. Not all fields are verified scenes. Literal overlap candidates, not proof of copying or absence of semantic duplication.'}


def summarize(report, ids, outputs, tag):
    phase = report['identity']['phase']
    assert phase in ('validation','final') and report['all_responses_complete']
    assert report['identity']['tag'] == tag
    assert len(ids) == len(set(ids)) == report['completed_pairs'] == len(report['pairs'])
    assert {row['id'] for row in report['pairs']} == set(ids)
    assert set(outputs) == {(number,variant) for number in ids for variant in ('original',tag)}
    scores,local,passed,utility = {},{},{},{}
    for row in report['pairs']:
        number = row['id']
        assert set(row['mapping']) == {'A','B'} and set(row['mapping'].values()) == {'original',tag}
        payload = {'moral':outputs[number,'original']['moral']}
        for label,variant in row['mapping'].items():
            payload[label] = {key:outputs[number,variant][key] for key in ['raw_text','ended_with_eos']}
        validate(row['result']['judgment'],payload)
        for label,variant in row['mapping'].items():
            key = (number,variant)
            scores[key] = row['result']['judgment'][label]
            story = outputs[key]
            local[key] = local_metrics(story['raw_text'],story['ended_with_eos'])
            values = [scores[key][criterion]['score'] for criterion in CRITERIA]
            passed[key] = int(local[key]['strict_format'] and local[key]['eos']
                and all(value is not None and value >= 3 for value in values))
            # This is an explicit reporting utility; null scores remain null in the underlying judgments.
            utility[key] = statistics.mean(values)/4 if all(value is not None for value in values) else 0.
    models = {}
    for variant in ('original',tag):
        stories = [outputs[number,variant] for number in ids]
        prefix = [repeated(complete_text_fields(story['raw_text'])) for story in stories]
        criteria = {}
        for criterion in CRITERIA:
            values = [scores[number,variant][criterion]['score'] for number in ids
                      if scores[number,variant][criterion]['score'] is not None]
            criteria[criterion] = {'valid_n':len(values),'mean_valid_only':statistics.mean(values) if values else None,
                'below_three_count':sum(value < 3 for value in values),'zero_count':sum(value == 0 for value in values)}
        ce = [story.get('teacher_forced') for story in stories]
        teacher = None
        if phase == 'validation':
            assert all(ce) and all(row['supervised_tokens'] > 0 and math.isfinite(row['ce']) for row in ce)
            tokens = sum(row['supervised_tokens'] for row in ce)
            value = sum(row['ce']*row['supervised_tokens'] for row in ce)/tokens
            teacher = {'stories':len(ce),'supervised_tokens':tokens,'token_weighted_ce':value,'conditional_perplexity':math.exp(value),
                'gold_prefix_EOS_top1_count':sum(row['gold_prefix_eos_rank']==1 for row in ce),
                'scope':'Gold assistant JSON and template ending, not free-generation EOS'}
        models[variant] = {'total_n':len(ids),'eos_count':sum(local[number,variant]['eos'] for number in ids),
            'strict_format_count':sum(local[number,variant]['strict_format'] for number in ids),
            'semantic_eligible_count':sum(local[number,variant]['semantic_eligible'] for number in ids),
            'quality_pass_count':sum(passed[number,variant] for number in ids),'quality_pass_denominator':len(ids),
            'failure_aware_content_utility':statistics.mean(utility[number,variant] for number in ids),
            'criteria':criteria,'teacher_forced':teacher,
            'prefix_repetition':{'with_any_complete_text_field':sum(row['complete_text_fields']>0 for row in prefix),
                'with_exact_repeat':sum(row['has_exact_repeat'] for row in prefix),
                'with_exact_repeat_among_nonEOS':sum(row['has_exact_repeat'] for row,story in zip(prefix,stories) if not story['ended_with_eos']),
                'scope':'Exact normalized text fields >=20chars; includes complete fields inside truncated JSON, not proof of semantic repetition'}}
    paired = {}
    for criterion in CRITERIA:
        differences = [scores[number,tag][criterion]['score']-scores[number,'original'][criterion]['score']
            for number in ids if all(scores[number,v][criterion]['score'] is not None for v in ('original',tag))]
        paired[criterion] = {'jointly_valid_n':len(differences),'mean_RL_minus_original':statistics.mean(differences) if differences else None,
            'bootstrap_95_interval':interval(differences)}
    def difference(values):
        return {'mean_RL_minus_original':statistics.mean(values),'bootstrap_95_interval':interval(values),'denominator':len(values)}
    preferences = Counter()
    examples = []
    for row in report['pairs']:
        judgment = row['result']['judgment']
        choice = judgment['preference'];preferences[row['mapping'].get(choice,choice)] += 1
        problems = []
        for label,variant in row['mapping'].items():
            for criterion,metric in judgment[label].items():
                if metric['score'] is None or metric['score'] < 3:
                    problems.append({'variant':variant,'criterion':criterion,**metric})
        if problems or judgment['human_review_needed']:
            examples.append({'id':row['id'],'criteria_needing_review':problems,
                'API_human_review_reason':judgment['human_review_reason'],'human_review_performed':False})
    return {'phase':phase,'policy_tag':tag,'prompt_count':len(ids),'models':models,'paired_criteria':paired,
        'quality_pass_difference':difference([passed[number,tag]-passed[number,'original'] for number in ids]),
        'failure_aware_utility_difference':difference([utility[number,tag]-utility[number,'original'] for number in ids]),
        'api_preferences':dict(preferences),'review_candidates':examples,
        'bootstrap':{'unit':'prompt ID, paired policies','repeats':4000,'seed':42,'multiple_criteria_adjusted':False},
        'selection_decision':'pending actual cycle diagnosis; never automatically adopt a model based on one score',
        'limitations':['One sampled story per prompt/policy; single training and decoding seed.',
            'Same API model family provides rewards and judging; no human/expert agreement established.',
            'Valid-only means and jointly valid pairs have different denominators from whole-set quality pass.',
            'N/A-as-zero reporting utility is separate from underlying nullable scores and the RL reward.',
            'No images generated; only text/visual-note consistency assessed.',
            'Final synthetic prompts are a held-out diagnostic sample, not a representative human-validated benchmark.']}


def calibration_sensitivity(report, outputs, tag, calibration_ids):
    assert report['identity']['phase'] == 'validation'
    ids = [row['id'] for row in report['pairs']]
    assert len(ids) == len(set(ids)) == 43
    excluded = set(calibration_ids)
    assert len(excluded) == len(calibration_ids) == 3 and excluded <= set(ids)
    kept = [number for number in ids if number not in excluded]
    assert len(kept) == 40
    subset = {**report, 'pairs': [row for row in report['pairs'] if row['id'] in kept], 'completed_pairs': 40}
    summary = summarize(subset, kept, {key: value for key, value in outputs.items() if key[0] in kept}, tag)
    summary['selection_decision'] = 'Secondary sensitivity only; primary43 remains the fixed development comparison. Do not select the favorable subset.'
    return {'primary_denominator': 43, 'excluded_technical_calibration_ids': sorted(excluded),
        'remaining_ids': kept, 'remaining_denominator': 40,
        'scope': 'Additional pre-update-declared diagnostic, not a changed split or a new independent test set. Technical API calibration used the excluded prompts, not policy-gradient training.',
        'original_in_A_count': sum(row['mapping']['A'] == 'original' for row in subset['pairs']),
        'summary': summary}


def historical_calibration(config, ids):
    start = Path(config.get('source_previous_run', OUT))
    if config.get('source_previous_run'):
        assert sha(start/'config.json') == config['source_previous_config_sha256']
    historical=next((source for source in run_lineage(start) if (source/'evaluation/calibration_split_audit.json').exists()),None)
    assert historical is not None, 'A preserved original calibration declaration is required'
    path = historical/'evaluation/calibration_split_audit.json'
    audit = p.read(path)
    assert audit['config_sha256'] == sha(historical/'config.json')
    assert audit['reward_protocol_frozen_sha256'] == sha(historical/'reward_protocol_frozen.json')
    assert audit['actual_RL_updates_at_declaration'] == 0 and audit['primary_denominator'] == 43
    assert audit['primary_validation_ids'] == ids
    calibration_ids = []
    for row in p.read(historical/'reward_protocol_frozen.json')['results']:
        source = Path(row['source'])
        assert row['source_sha256'] == sha(source)
        pilot = p.read(source)
        assert pilot['state'] == 'completed' and pilot['job']['split'] == 'pilot'
        calibration_ids.append(str(pilot['job']['source_id']))
    assert calibration_ids == audit['technical_calibration_ids']
    assert len(calibration_ids) == len(set(calibration_ids)) == 3 and set(calibration_ids) <= set(ids)
    return path, calibration_ids


def training_output_sources(root, tag):
    batches=committed_batches(root)
    assert tag in batches, 'Only a committed policy can define its training references'
    config=p.read(root/'config.json')
    selected={batch for name,batch in batches.items() if int(name.split('_')[1])<=int(tag.split('_')[1])}
    paths=[path for batch in sorted(selected) for path in sorted((batch/'rollouts').glob('*.json'))]
    for path in paths:
        number=p.read(path)['identity']['source_id']
        assert number in config['train_ids'] and number not in config['validation_ids']+config['excluded_test_ids']
    return paths


def plot_comparison(summary, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    tag = summary['policy_tag']
    models = [summary['models'][variant] for variant in ('original',tag)]
    fig,axes = plt.subplots(1,3,figsize=(16,5))
    if all(model['teacher_forced'] for model in models):
        axes[0].bar(['원본',tag],[model['teacher_forced']['token_weighted_ce'] for model in models],color=['#5D7784','#E9542E'])
        axes[0].set_ylabel('정답 토큰 가중 CE')
        axes[0].set_title('무절단 정답 CE / 자유 생성 품질과 구분')
    else:
        axes[0].text(.5,.5,'새 교훈에는 정답 이야기 없음\nCE 산출 대상 아님',ha='center',va='center')
        axes[0].set_axis_off()
    keys = ['eos_count','strict_format_count','quality_pass_count']
    for i,(model,color) in enumerate(zip(models,['#5D7784','#E9542E'])):
        axes[1].bar([x+(i-.5)*.34 for x in range(3)],
            [model[key]/model['total_n'] for key in keys],width=.34,color=color,label=['원본',tag][i])
    axes[1].set_xticks(range(3),['EOS','엄격 형식','전체 품질'])
    axes[1].set_ylim(0,1.05);axes[1].set_title(f"실패 포함 전체{summary['prompt_count']}개 분모");axes[1].legend()
    labels = ['교훈','인과','감정','낭독','장면 밀도','글·그림','연령']
    for y,criterion in enumerate(CRITERIA):
        row = summary['paired_criteria'][criterion]
        value = row['mean_RL_minus_original']
        if value is None:
            axes[2].text(0,y,'N/A',va='center')
        else:
            ci = row['bootstrap_95_interval']
            if ci:
                axes[2].hlines(y,ci[0],ci[1],color='#E9542E',linewidth=2)
            axes[2].scatter([value],[y],color='#E9542E')
    axes[2].axvline(0,color='#809099',linewidth=.8)
    axes[2].set_yticks(range(7),[f"{label} (N={summary['paired_criteria'][criterion]['jointly_valid_n']})" for label,criterion in zip(labels,CRITERIA)])
    axes[2].invert_yaxis();axes[2].set_xlabel('RL − 원본 / paired bootstrap95%')
    axes[2].set_title('API 내용 점수: 양쪽 유효 작품만')
    fig.suptitle(f"{summary['policy_tag']} · {summary['phase']} · 단일 seed / API 의견 / 사람 검토 미실시")
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    with (folder/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.writer(stream);writer.writerow(['metric','original','RL','original_valid_N','RL_valid_N'])
        for key in keys:
            writer.writerow([key,models[0][key],models[1][key],models[0]['total_n'],models[1]['total_n']])
        for criterion in CRITERIA:
            one,two = [model['criteria'][criterion] for model in models]
            writer.writerow([criterion,one['mean_valid_only'],two['mean_valid_only'],one['valid_n'],two['valid_n']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['validation','final'],default='validation')
    parser.add_argument('--policy-tag',default='update_0001')
    args = parser.parse_args()
    folder = OUT/f'evaluation/quality/{args.phase}_{args.policy_tag}'
    report = p.read(folder/'report.json')
    pairs = source_pairs(args.phase,args.policy_tag)
    verify_generated_sources(args.phase,args.policy_tag,pairs)
    outputs = {}
    for number,before,after in pairs:
        assert report['identity']['source_sha256'][str(before)] == sha(before)
        assert report['identity']['source_sha256'][str(after)] == sha(after)
        outputs[number,'original'] = p.read(before)
        outputs[number,args.policy_tag] = p.read(after)
    summary = summarize(report,[row[0] for row in pairs],outputs,args.policy_tag)
    config = p.read(OUT/'config.json')
    if args.phase == 'validation':
        audit_path, calibration_ids = historical_calibration(config, [row[0] for row in pairs])
        summary['technical_calibration_sensitivity'] = calibration_sensitivity(report, outputs, args.policy_tag, calibration_ids)
        summary['technical_calibration_sensitivity']['pre_update_declaration_sha256'] = sha(audit_path)
        summary['technical_calibration_sensitivity']['historical_declaration'] = str(audit_path)
    selected_batch = committed_batch(OUT, args.policy_tag)
    summary['committed_training_diagnostics'] = str(write_training_report(batch=selected_batch, output=folder/'training_report'))
    development = [row for row in p.read(p.PROCESSED/'dataset.json') if row['source_id'] in config['train_ids']]
    references = [{'id':'v5_'+str(row['source_id']),'raw_text':json.dumps(row['output'],ensure_ascii=False)} for row in development]
    training_sources = training_output_sources(OUT, args.policy_tag)
    references += [{'id':'on_policy_'+str(index)+'_'+str(row['identity']['source_id']),'raw_text':row['raw_text']}
        for index,row in enumerate(p.read(path) for path in training_sources)]
    summary['literal_overlap_check'] = copy_candidates(references,outputs)
    summary['literal_overlap_check']['reference_scope'] = 'v5 train200 gold books and committed training outputs up to the selected update; no validation/test61/final prompts. Gold v5 stories were not supervised targets in this RL run.'
    summary['literal_overlap_check']['training_output_sha256'] = {str(path):sha(path) for path in training_sources}
    summary['API_ledger_scope'] = 'All new calibration/reward/quality calls share the same budget_ledger.json; this report makes no new paid request.'
    save(folder/'summary.json',summary)
    plot_comparison(summary,folder)
    lines = [f'# {args.policy_tag}: {args.phase} 원본/RL 비교','',f'전체 분모: {len(pairs)}교훈. API 의견이며 사람/전문가 인증이 아니다.','',
        '| 지표 | 원본 | RL |','|---|---:|---:|']
    for label,key in [('EOS','eos_count'),('엄격 형식','strict_format_count'),('전체 품질 통과','quality_pass_count'),('의미 평가 가능','semantic_eligible_count')]:
        lines.append(f"| {label} | {summary['models']['original'][key]}/{len(pairs)} | {summary['models'][args.policy_tag][key]}/{len(pairs)} |")
    lines += ['', '7항목별 유효 N·점수·차이·paired bootstrap 구간, 실패 원문 근거와 한계는 summary.json에 보존한다.',
        '미종료/파싱 실패도 전체 분모에 포함한다. 문자 반복/동일 구절은 검토 후보이며 새로움 또는 암기의 확정 판정이 아니다.',
        '현재 결과만으로 다음 모델 선택이나 보상 개정을 자동 확정하지 않는다.']
    if args.phase == 'validation':
        lines += ['', '주 비교는 전체43개다. 기술교정에 사용한3교훈을 제외한40개 보조 민감도 분석은 summary.json의 technical_calibration_sensitivity에 있다.',
            '43개/40개에서 유리한 쪽만 선택하지 않으며,40개도 새로운 독립 최종 평가셋으로 소개하지 않는다.']
    (folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'phase':args.phase,'prompts':len(pairs),'summary':str(folder/'summary.json'),'new_API_calls':0}))


if __name__=='__main__':main()
