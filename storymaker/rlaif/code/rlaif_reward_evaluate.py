"""Evaluate saved stories using the accepted training reward and train-only RAG."""
import argparse
import csv
import hashlib
import json
import math
import msvcrt
import os
import statistics
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import yaml
from storymaker import project as p
from storymaker.qwen.code.evaluation_report import interval
from storymaker.qwen.code.prefix_diagnostics import complete_text_fields, repeated
from storymaker.rlaif.code import rlaif_reward_api as reward
from storymaker.rlaif.code import reward_rag as rag
from storymaker.rlaif.code.rlaif_budget import lock_path, ledger_path
from storymaker.rlaif.code.rlaif_environment import OUT
from storymaker.rlaif.code.rlaif_evaluate import local_metrics, sha
from storymaker.rlaif.code.rlaif_quality_api import source_pairs, verify_generated_sources
from storymaker.rlaif.code.rlaif_quality_report import copy_candidates, training_output_sources, historical_calibration


DIAGNOSTICS=('novelty','completion',*reward.CRITERIA)


def verified_result(source, destination, context):
    return reward.verified_score(source,destination,context)


def accepted_protocol(config):
    if config.get('reward_protocol_kind') == 'fixed_criteria_diagnosis_then_direct_score':
        from storymaker.rlaif.code import rlaif_reward_pipeline as pipeline
        return pipeline.accepted_protocol(config)
    frozen=p.read(OUT/'reward_protocol_frozen.json')
    assert frozen['api_code_sha256']==sha(Path(reward.__file__))
    if frozen.get('temporal_review_required'):
        from storymaker.rlaif.code import rlaif_temporal_review as review
        assert frozen['temporal_review_module_sha256']==sha(Path(review.__file__))
    assert frozen['model']==config['api_model']
    acceptance=Path(frozen['acceptance_file'])
    assert sha(acceptance)==frozen['acceptance_sha256']
    source_config=acceptance.parent.parent/'config.json'
    assert sha(source_config)==frozen['config_sha256']
    old=p.read(source_config)
    for key in ('train_ids','validation_ids','excluded_test_ids','api_model','rag_required'):
        assert old[key]==config[key]
    assert reward.check_calibration(old,acceptance.parent)['passed'], 'Reward contrasts must pass before model evaluation'
    return frozen


def summarize(rows, tag, phase):
    assert phase in ('validation','final')
    ids=[row['id'] for row in rows]
    assert len(ids)==len(set(ids)) and rows
    models={};paired={}
    for variant in ('original',tag):
        records=[row['models'][variant] for row in rows]
        assert all(record['result']['certified_decimal_places']==6 for record in records)
        metrics={name:[] for name in DIAGNOSTICS}
        for record in records:
            result=record['result'];judgment=result['judgment']
            assert type(record['local']['semantic_eligible']) is bool
            assert not record['local']['semantic_eligible'] or record['local']['eos']
            assert -1<=result['normalized_reward']<=1 and type(result['selected_score']) is int
            for name in DIAGNOSTICS:
                value=(judgment[name] if name in ('novelty','completion') else judgment['content'][name])['score']
                assert type(value) is int and 0<=value<=4
                metrics[name].append(value)
        models[variant]={'total_n':len(rows),
            'mean_expected_reward':statistics.mean(record['result']['normalized_reward'] for record in records),
            'mean_selected_score':statistics.mean(record['result']['selected_score'] for record in records),
            'diagnostics':{name:{'mean':statistics.mean(values),'denominator':len(rows),'below_three':sum(value<3 for value in values)} for name,values in metrics.items()},
            'eos_count':sum(record['local']['eos'] for record in records),
            'strict_format_count':sum(record['local']['strict_format'] for record in records),
            'exact_prefix_repeat_count':sum(record['prefix_repetition']['has_exact_repeat'] for record in records),
            'high_reward_anchor_count':sum(record['local']['eos'] and record['local']['strict_format'] and record['result']['selected_score']>=8
                and all(metrics[name][index]>=3 for name in DIAGNOSTICS) for index,record in enumerate(records))}
        eligible=[index for index,record in enumerate(records) if record['local']['semantic_eligible']]
        models[variant].update(semantic_eligible_n=len(eligible),semantic_ineligible_n=len(rows)-len(eligible),
            diagnostics_scope='All-output API opinions, including failed/unreadable outputs; not an eligible-only content mean.',
            eligible_content_diagnostics={name:{'mean':statistics.mean(metrics[name][index] for index in eligible) if eligible else None,
                'denominator':len(eligible)} for name in reward.CRITERIA},
            failure_inclusive_content_utility=statistics.mean(
                sum(metrics[name][index] for name in reward.CRITERIA)/(4*len(reward.CRITERIA)) if index in eligible else 0
                for index in range(len(rows))))
        ce=[record.get('teacher_forced') for record in records]
        available=[value for value in ce if value is not None]
        auxiliary={'available_n':len(available),'expected_n':len(rows),'token_weighted_ce':None,'role':'Reference-target diagnostic, excluded from reward ranking.'}
        if phase=='validation' and len(available)==len(rows):
            assert all(value['supervised_tokens']>0 and math.isfinite(value['ce']) for value in available)
            auxiliary['token_weighted_ce']=sum(value['ce']*value['supervised_tokens'] for value in available)/sum(value['supervised_tokens'] for value in available)
        models[variant]['auxiliary_reference_CE']=auxiliary
    for name in ('expected_reward',*DIAGNOSTICS):
        differences=[]
        for row in rows:
            values=[]
            for variant in ('original',tag):
                result=row['models'][variant]['result'];judgment=result['judgment']
                value=result['normalized_reward'] if name=='expected_reward' else (
                    judgment[name] if name in ('novelty','completion') else judgment['content'][name])['score']
                values.append(value)
            differences.append(values[1]-values[0])
        paired[name]={'mean_RL_minus_original':statistics.mean(differences),'bootstrap_95_interval':interval(differences),'denominator':len(rows)}
    paired_eligible={}
    for name in reward.CRITERIA:
        differences=[row['models'][tag]['result']['judgment']['content'][name]['score']-row['models']['original']['result']['judgment']['content'][name]['score']
            for row in rows if all(row['models'][variant]['local']['semantic_eligible'] for variant in ('original',tag))]
        paired_eligible[name]={'mean_RL_minus_original':statistics.mean(differences) if differences else None,
            'bootstrap_95_interval':interval(differences) if differences else None,'denominator':len(differences)}
    utility_difference=[]
    for row in rows:
        utilities=[sum(record['result']['judgment']['content'][name]['score'] for name in reward.CRITERIA)/(4*len(reward.CRITERIA))
            if record['local']['semantic_eligible'] else 0 for record in (row['models']['original'],row['models'][tag])]
        utility_difference.append(utilities[1]-utilities[0])
    delta=[row['models'][tag]['result']['normalized_reward']-row['models']['original']['result']['normalized_reward'] for row in rows]
    return {'phase':phase,'policy_tag':tag,'prompt_count':len(rows),'models':models,'paired':paired,
        'paired_eligible_content':paired_eligible,
        'paired_failure_inclusive_content_utility':{'mean_RL_minus_original':statistics.mean(utility_difference),
            'bootstrap_95_interval':interval(utility_difference),'denominator':len(rows),
            'scope':'Secondary utility only: EOS+readable JSON scene text gets mean seven content scores/4; ineligible gets0. Raw API reward is unchanged.'},
        'reward_wins_ties_losses':{'win':sum(value>0 for value in delta),'tie':sum(value==0 for value in delta),'loss':sum(value<0 for value in delta)},
        'primary_metric':'Expected reward from the accepted score-token likelihoods; nine diagnostic scores and whole-denominator technical failures.',
        'limitations':['Same API and rubric provide training reward and evaluation: this measures agreement with that reward, not independent expert truth.',
            'Rewarding novelty uses retrieved train200 references, not worldwide originality certification.',
            'One sampled story per prompt and a single decoding seed; selected score differs from likelihood expectation.',
            'All failures remain in the denominator; no quality rerolls and no scene-level sample inflation.']}


def calibration_sensitivity(rows, tag, calibration_ids):
    ids=[row['id'] for row in rows]
    excluded=set(calibration_ids)
    assert len(ids)==len(set(ids))==43
    assert len(excluded)==len(calibration_ids)==3 and excluded<=set(ids)
    remaining=[row for row in rows if row['id'] not in excluded]
    assert len(remaining)==40
    return {'primary_denominator':43,'excluded_technical_calibration_ids':sorted(excluded),
        'remaining_denominator':40,'remaining_ids':[row['id'] for row in remaining],
        'summary':summarize(remaining,tag,'validation'),
        'scope':'Pre-update-declared secondary sensitivity; retain the primary43 and never select the favorable subset. The40 are not a new independent test set.'}


def write_summary(summary, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tag=summary['policy_tag'];variants=('original',tag);models=[summary['models'][variant] for variant in variants]
    folder.mkdir(parents=True,exist_ok=True);reward.save(folder/'summary.json',summary)
    rows=[['expected_reward',*(model['mean_expected_reward'] for model in models)]]
    rows += [[name,*(model['diagnostics'][name]['mean'] for model in models)] for name in DIAGNOSTICS]
    rows += [[name,*(model[key]/model['total_n'] for model in models)] for name,key in (
        ('EOS_rate','eos_count'),('JSON_rate','strict_format_count'),('exact_repeat_rate','exact_prefix_repeat_count'))]
    rows += [[name,*(model[name] for model in models)] for name in ('semantic_eligible_n','failure_inclusive_content_utility')]
    rows += [['eligible_'+name,*(model['eligible_content_diagnostics'][name]['mean'] for model in models)] for name in reward.CRITERIA]
    if all('literal_overlap_candidate_count' in model for model in models):
        rows.append(['literal_overlap_candidate_rate',*(model['literal_overlap_candidate_count']/model['total_n'] for model in models)])
    rows.append(['auxiliary_reference_CE',*(model['auxiliary_reference_CE']['token_weighted_ce'] for model in models)])
    with (folder/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['metric','original',tag]);writer.writerows(rows)
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(16,6.5),gridspec_kw={'width_ratios':[.9,1.65,1.05]})
    colors=['#82959D','#E9542E']
    axes[0].bar(['원본',tag],[model['mean_expected_reward'] for model in models],color=colors)
    axes[0].axhline(0,color='#819098',linewidth=.8);axes[0].set_ylim(-1,1)
    axes[0].set_title('주 지표 · 보상 확률 기댓값');axes[0].set_ylabel('정규화 보상 -1~1')
    labels=['새로움','완결성','교훈 구현','인과관계','감정 연결','한국어 낭독','장면 밀도','글·그림 설명','연령 적합성']
    for index,name in enumerate(DIAGNOSTICS):
        paired=summary['paired'][name];ci=paired['bootstrap_95_interval']
        if ci:axes[1].hlines(index,ci[0],ci[1],color=colors[1],linewidth=2)
        axes[1].scatter([paired['mean_RL_minus_original']],[index],color=colors[1])
    axes[1].set_yticks(range(9),labels);axes[1].invert_yaxis();axes[1].axvline(0,color='#819098',linewidth=.8)
    axes[1].set_title('전체 API 진단 차이 · 작품 단위95% 구간');axes[1].set_xlabel('RL - 원본 · 0~4점 척도')
    for index,model in enumerate(models):
        axes[2].bar([x+index*.35 for x in range(3)],[model[key]/model['total_n'] for key in ('eos_count','strict_format_count','exact_prefix_repeat_count')],width=.35,color=colors[index],label=variants[index])
    axes[2].set_xticks([.175,1.175,2.175],['EOS','JSON','완전 반복']);axes[2].set_ylim(0,1)
    axes[2].set_title(f'기술 지표 · 전체{summary["prompt_count"]}편 분모');axes[2].legend()
    fig.suptitle(f'{summary["phase"]} · 학습과 동일한 보상 기준 비교',fontsize=17)
    fig.text(.5,.015,f'내용 평가가능 N={models[0]["semantic_eligible_n"]}/{models[1]["semantic_eligible_n"]} · 실패포함 내용효용={models[0]["failure_inclusive_content_utility"]:.3f}/{models[1]["failure_inclusive_content_utility"]:.3f} · 동일 API 평가 · CE는 보조 지표',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.94]);fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['validation','final'],default='validation')
    parser.add_argument('--policy-tag',default='update_0001')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--prepare-rag',action='store_true')
    action.add_argument('--execute',action='store_true')
    args=parser.parse_args();config=p.read(OUT/'config.json')
    pairs=source_pairs(args.phase,args.policy_tag)
    folder=OUT/f'evaluation/reward_metrics/{args.phase}_{args.policy_tag}'
    sources={variant:[row[index] for row in pairs] for variant,index in [('original',1),(args.policy_tag,2)]}
    if not args.prepare_rag and not args.execute:
        folder.mkdir(parents=True,exist_ok=True)
        reward.save(folder/'plan.json',{'phase':args.phase,'tag':args.policy_tag,'prompts':len(pairs),
            'maximum_story_requests':len(pairs)*2,'exact_request_reuse':True,'criteria':list(DIAGNOSTICS),
            'primary':'Accepted training reward probability expectation; CE remains auxiliary.',
            'RAG':'Train200 only; no own held-out reference.','new_paid_calls':0,'execute':False})
        print(json.dumps({'plan_only':True,'prompts':len(pairs),'maximum_story_requests':len(pairs)*2}));return
    assert p.read(OUT/'control.json')['enabled']
    verify_generated_sources(args.phase,args.policy_tag,pairs)
    if args.prepare_rag:
        for variant,paths in sources.items():reward.prepare_rag(config,folder/variant,paths,include_source=False)
        return
    frozen=accepted_protocol(config)
    paths=[path for values in sources.values() for path in values]
    contexts={str(path):p.read(folder/variant/'rag'/path.name) for variant,values in sources.items() for path in values}
    for path in paths:
        context=contexts[str(path)]
        assert context['source_sha256']==sha(path) and context['corpus_sha256']==rag.sha(rag.FOLDER/'corpus.jsonl')
        assert context['train_only'] is True and context['include_training_source'] is False
        reference_ids={reference['source_id'] for reference in context['references'] if reference['source_id'] is not None}
        assert reference_ids<=set(config['train_ids'])
    identity={'phase':args.phase,'tag':args.policy_tag,'config_sha256':sha(OUT/'config.json'),
        'protocol':frozen,'source_sha256':{str(path):sha(path) for path in paths},
        'code_sha256':{name:sha(p.code_path(name)) for name in ['rlaif_reward_evaluate.py','rlaif_reward_api.py','reward_rag.py','rlaif_budget.py']},
        'context_sha256':{str(folder/variant/'rag'/path.name):sha(folder/variant/'rag'/path.name) for variant,values in sources.items() for path in values}}
    if config.get('reward_protocol_kind') == 'fixed_criteria_diagnosis_then_direct_score':
        identity['code_sha256']['rlaif_reward_pipeline.py']=sha(p.code_path('rlaif_reward_pipeline.py'))
    manifest=folder/'manifest.json'
    if manifest.exists():assert p.read(manifest)==identity
    else:reward.save(manifest,identity)
    key=yaml.safe_load((p.ROOT / 'secrets/api_key.yaml').read_text('utf-8'))['openai_api_key']
    rows=[]
    try:
        with lock_path(OUT).open('a+b') as lock:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            before_ledger=p.read(ledger_path(OUT))
            for index,(number,before,after) in enumerate(pairs):
                assert p.read(OUT/'control.json')['enabled']
                assert sha(OUT/'config.json')==identity['config_sha256']
                assert all(sha(p.code_path(name))==digest for name,digest in identity['code_sha256'].items())
                row={'id':number,'models':{}}
                order=[('original',before),(args.policy_tag,after)]
                if index%2:order.reverse()
                for variant,path in order:
                    assert sha(path)==identity['source_sha256'][str(path)]
                    assert sha(folder/variant/'rag'/path.name)==identity['context_sha256'][str(folder/variant/'rag'/path.name)]
                    destination=folder/variant/'responses'/number
                    context=contexts[str(path)]
                    if frozen.get('temporal_review_required'):
                        context=reward.prepare_temporal_context(path,folder/variant,key,context,evaluation_phase=args.phase)
                    reward.score(path,destination,key,context,evaluation_phase=args.phase)
                    result=verified_result(path,destination,context)
                    story=p.read(path)
                    row['models'][variant]={'source':str(path),'source_sha256':sha(path),'result':result,
                        'local':local_metrics(story['raw_text'],story['ended_with_eos']),
                        'teacher_forced':story.get('teacher_forced'),
                        'prefix_repetition':repeated(complete_text_fields(story['raw_text']))}
                rows.append(row)
                reward.save(folder/'status.json',{'stage':'scoring','completed_pairs':len(rows),'total_pairs':len(pairs),'pid':os.getpid(),'at_utc':datetime.now(timezone.utc).isoformat()})
            after_ledger=p.read(ledger_path(OUT))
        assert len(rows)==(43 if args.phase=='validation' else 32)
        cost=lambda ledger:Decimal(str(ledger.get('previous_RL_spend_usd',0)))+sum((Decimal(row['cost_upper_usd']) for row in ledger['calls']),Decimal(0))
        reward.save(folder/'report.json',{'identity':identity,'rows':rows,'all_responses_complete':True,
            'new_paid_requests':len(after_ledger['calls'])-len(before_ledger['calls']),
            'new_cost_upper_usd':str(cost(after_ledger)-cost(before_ledger))})
        summary=summarize(rows,args.policy_tag,args.phase)
        if args.phase=='validation':
            audit_path,calibration_ids=historical_calibration(config,[row['id'] for row in rows])
            summary['technical_calibration_sensitivity']=calibration_sensitivity(rows,args.policy_tag,calibration_ids)
            summary['technical_calibration_sensitivity'].update(
                historical_declaration=str(audit_path),pre_update_declaration_sha256=sha(audit_path))
        development=[book for book in p.read(p.PROCESSED/'dataset.json') if book['source_id'] in config['train_ids']]
        references=[{'id':'v5_'+str(book['source_id']),'raw_text':json.dumps(book['output'],ensure_ascii=False)} for book in development]
        training_sources=training_output_sources(OUT,args.policy_tag)
        references += [{'id':'trained_'+str(index),'raw_text':p.read(path)['raw_text']} for index,path in enumerate(training_sources)]
        outputs={(row['id'],variant):p.read(row['models'][variant]['source']) for row in rows for variant in ('original',args.policy_tag)}
        summary['literal_overlap_check']=copy_candidates(references,outputs)
        summary['literal_overlap_check']['training_source_sha256']={str(path):sha(path) for path in training_sources}
        for variant,model in summary['models'].items():
            model['literal_overlap_candidate_count']=len({match['id'] for match in summary['literal_overlap_check']['matches'] if match['variant']==variant})
        write_summary(summary,folder)
        reward.save(folder/'status.json',{'stage':'complete','completed_pairs':len(rows),'total_pairs':len(pairs),'actual_training_updates':0})
    except BaseException as error:
        reward.save(folder/'status.json',{'stage':'needs_attention','error':type(error).__name__,'message':str(error),'automatic_retry':False})
        raise


if __name__=='__main__':main()
