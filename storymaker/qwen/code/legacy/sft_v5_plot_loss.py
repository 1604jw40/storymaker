"""Snapshot existing v5 loss logs and plot them on CPU without touching training."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

from storymaker.qwen.code.legacy import sft_v5_prepare as p
def main():
    now=datetime.now(timezone.utc)
    folder=p.RUN/'analysis'/('loss_'+now.strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir(parents=True,exist_ok=False)
    status=p.read(p.RUN/'status.json')
    config_digest=hashlib.sha256((p.RUN/'config.json').read_bytes()).hexdigest()
    stdout=(p.RUN/'stdout.log').read_text('utf-8')
    live=[]
    for line in stdout.splitlines():
        if not line.startswith('{'):continue
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        if event.get('state')=='training':live.append(event)
    folds=[]
    for number in range(1,6):
        root=p.RUN/'trial_0000'/f'fold_{number}'
        complete=(root/'result.json').exists()
        states=[root/'trainer_state.json'] if complete else sorted(root.glob('checkpoint-*/trainer_state.json'),key=lambda x:int(x.parent.name.split('-')[-1]))
        assert states,root
        state_path=states[-1]
        source=state_path.read_bytes();state=json.loads(source.decode('utf-8-sig'))
        rows={}
        for entry in state['log_history']:
            for field,kind in [('loss','train'),('eval_loss','validation')]:
                if field in entry:rows[(kind,entry['step'])]={'kind':kind,'step':entry['step'],'epoch':entry['epoch'],'ce':entry[field]}
        # Checkpoint history is authoritative; only append the active worker's newer logs.
        if not complete:
            for event in live:
                if event.get('stage')!=f'trial_0000/fold_{number}' or event.get('pid')!=status['pid']:continue
                if event.get('optimizer_step',0)<=state['global_step']:continue
                for field,kind in [('loss','train'),('eval_loss','validation')]:
                    if field in event.get('logs',{}):
                        rows[(kind,event['optimizer_step'])]={'kind':kind,'step':event['optimizer_step'],'epoch':event['epoch'],'ce':event['logs'][field]}
        records=sorted(rows.values(),key=lambda x:(x['step'],x['kind']))
        assert len(records)==len({(r['kind'],r['step']) for r in records})
        assert all(math.isfinite(r['ce']) and r['ce']>0 and 0<r['epoch']<=5 for r in records)
        training=[r for r in records if r['kind']=='train'];validation=[r for r in records if r['kind']=='validation']
        if complete:
            result=p.read(root/'result.json')
            assert len(validation)==5 and [r['epoch'] for r in validation]==[1,2,3,4,5]
            np.testing.assert_allclose([r['ce'] for r in validation],result['eval_loss_by_epoch'],rtol=0,atol=1e-9)
        increases=[{'from_epoch':a['epoch'],'to_epoch':b['epoch'],'increase':b['ce']-a['ce']} for a,b in zip(validation,validation[1:]) if b['ce']>a['ce']]
        folds.append({'fold':number,'complete':complete,'source':str(state_path),'source_sha256':hashlib.sha256(source).hexdigest(),'train':training,'validation':validation,'validation_increases':increases})
    finished=[f for f in folds if f['complete']]
    assert len(finished)>=1
    epochs=np.arange(1,6)
    matrix=np.array([[r['ce'] for r in f['validation']] for f in finished])
    means=matrix.mean(axis=0)
    assert means.shape==(5,) and np.isfinite(means).all()
    snapshot={'captured_at_utc':now.isoformat(),'captured_at_kst':now.astimezone(timezone(timedelta(hours=9))).isoformat(),
        'trial':0,'learning_rate':8e-5,'status':status,'config_sha256':config_digest,
        'folds':folds,'fixed_cohort_mean_validation':{'fold_ids':[f['fold'] for f in finished],'epochs':epochs.tolist(),'ce':means.tolist(),'successive_improvements':(-np.diff(means)).tolist()},
        'validation_rise_intervals':sum(len(f['validation_increases']) for f in folds),
        'method':'Completed trainer_state histories; latest active checkpoint plus newer active-worker stdout. No restarts or smoke runs mixed. Train is logged minibatch CE with dropout; validation is full held-out evaluation with dropout disabled. Smoothing is descriptive, not an epoch train-set evaluation.'}
    p.save(folder/'snapshot.json',snapshot)
    with (folder/'loss_points.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['trial','fold','complete','kind','step','epoch','ce']);writer.writeheader()
        for f in folds:
            for row in f['train']+f['validation']:writer.writerow({'trial':0,'fold':f['fold'],'complete':f['complete'],**row})
    font_manager.fontManager.addfont(r'C:\Windows\Fonts\malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'axes.titlesize':13,'axes.labelsize':11})
    fig,axes=plt.subplots(2,3,figsize=(15.5,9.3))
    blue,orange,purple='#2563A6','#DD692C','#7156A5'
    values=[r['ce'] for f in folds for r in f['train']+f['validation']]
    limits=(min(values)-.025,max(values)+.025)
    for ax,f in zip(axes.flat,folds):
        x=np.array([r['epoch'] for r in f['train']]);y=np.array([r['ce'] for r in f['train']])
        smooth=np.array([y[max(0,i-4):i+1].mean() for i in range(len(y))])
        ax.plot(x,y,color=blue,alpha=.20,lw=.9)
        ax.plot(x,smooth,color=blue,lw=2)
        vx=[r['epoch'] for r in f['validation']];vy=[r['ce'] for r in f['validation']]
        ax.plot(vx,vy,'o-',color=orange,lw=2.3,ms=5)
        ax.annotate(f'{vy[-1]:.4f}',(vx[-1],vy[-1]),xytext=(-4,10),textcoords='offset points',ha='right',color=orange,fontsize=10)
        ax.set(title=f'Fold {f["fold"]} · '+('5에포크 완료' if f['complete'] else '학습 진행 중'),xlim=(0,5.18),ylim=limits,xlabel='에포크',ylabel='Cross-entropy loss')
        ax.set_xticks(range(6))
    ax=axes.flat[-1]
    ax.fill_between(epochs,matrix.min(axis=0),matrix.max(axis=0),color=purple,alpha=.12,label='fold 간 최소~최대')
    ax.plot(epochs,means,'o-',color=purple,lw=2.5,ms=6,label=f'동일한 완료 {len(finished)}개 fold 평균')
    for x,y in zip(epochs,means):ax.annotate(f'{y:.4f}',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
    ax.set(title=f'완료 {len(finished)}개 fold 평균 검증 CE · 확대',xlim=(.7,5.3),ylim=(matrix.min()-.015,matrix.max()+.025),xlabel='에포크',ylabel='Validation cross-entropy')
    ax.set_xticks(epochs);ax.legend(loc='upper right',fontsize=9,frameon=False)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('v5 SFT 학습·검증 loss 추이',fontsize=20,fontweight='bold',y=.975)
    stamp=now.astimezone(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
    fig.text(.5,.925,f'첫 학습률 후보 8e-5  |  {stamp}  |  완료 fold {len(finished)}/5',ha='center',fontsize=12,color='#4B5563')
    handles=[Line2D([0],[0],color=blue,lw=2,label='학습 CE · 최근 5회 로그 이동평균'),Line2D([0],[0],color=blue,alpha=.25,lw=1,label='학습 CE · 원래 로그'),Line2D([0],[0],color=orange,marker='o',lw=2,label='검증 CE · 에포크별 전체 검증셋')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.90),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.035,'각 fold는 새 모델로 시작한 별도 학습입니다. 평균 검증선은 같은 완료 fold만 사용합니다. 진행 중인 fold는 평균에 포함하지 않습니다.\n학습 CE는 미니배치 로그이고 검증 CE는 전체 검증 평균이므로, 두 값의 차이 자체를 과적합 지표로 단정하지 않습니다.',ha='center',fontsize=10,color='#4B5563',linespacing=1.6)
    fig.subplots_adjust(top=.82,bottom=.14,hspace=.36,wspace=.25)
    fig.savefig(folder/'loss_curves.png',dpi=180,facecolor='white')
    fig.savefig(folder/'loss_curves.svg',facecolor='white')
    plt.close(fig)
    assert hashlib.sha256((p.RUN/'config.json').read_bytes()).hexdigest()==config_digest
    print(json.dumps({'folder':str(folder),'png':str(folder/'loss_curves.png'),'completed_folds':len(finished),'mean_validation':means.tolist(),'mean_improvements':(-np.diff(means)).tolist(),'validation_rise_intervals':snapshot['validation_rise_intervals'],'active_validation':folds[-1]['validation']},ensure_ascii=False))


if __name__=='__main__':main()
