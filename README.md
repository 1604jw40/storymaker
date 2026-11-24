# Storymaker · 교훈 기반 한국어 그림책 생성

교훈 한 줄을 받아 5~7세가 읽을 수 있는 새로운 그림책 이야기를 생성합니다. 인물, 장면 본문, 그림 설명을 JSON으로 출력하며, 첫 출력부터 이야기와 형식이 갖춰진 작품의 비율을 높이는 것이 목표입니다.

## 프로젝트 구조

| 폴더 | 역할 |
|---|---|
| [storymaker/data_augmentation](storymaker/data_augmentation/code) | 원작 데이터 검증, API 각색, v5 데이터 정리 |
| [storymaker/qwen](storymaker/qwen/code) | 원본 Qwen 비교, LoRA SFT, CE와 실제 생성 평가 |
| [storymaker/rlaif](storymaker/rlaif/code) | 훈련 자료 검색 RAG, GPT-4.1 직접 보상, 정책·가치 모델 갱신 |
| [reports](reports) | 실험 설정, 테스트 집계, 학습 그래프, 후속 보상 검증 |

## 실험 흐름과 결과

1. 수집한 우화에서 교훈이 있는 304편을 구성하고, API로 인물·장면·그림 설명을 갖춘 데이터로 정리했습니다.
2. 개발 243편으로 Qwen3.5-4B의 LoRA 설정을 탐색해 5epoch 재학습했습니다. 참조 정답 CE는 1.3506→1.1484로 개선됐고, 고정 테스트 61편의 정상 종료는 60→39편으로 줄었습니다. [SFT 탐색 손실](reports/loss_hpo_grid.png) · [최종 재학습 손실](reports/loss_final_refit.png)
3. 원본 Qwen부터 이어온 정책4를 출발점으로, 훈련 교훈 200개를 16편씩 12회와 마지막 8편으로 순회했습니다. GPT-4.1 점수 토큰의 기댓값을 보상으로 쓰고, REINFORCE·학습된 상태 가치 기준선·고정 참조 KL로 정책과 가치를 각각 13회 갱신했습니다. [실험 설정](reports/experiment-settings.json) · [배치별 보상과 가치 기준선](reports/rlaif-batch-reward-lines.png)

### 독립 테스트 32개 교훈

| 지표 | 원본 Qwen | 출발 정책4 | 본학습 종료 update18 |
|---|---:|---:|---:|
| 엄격 JSON 통과 | 29/32 | 31/32 | 30/32 |
| EOS 종료 | 29/32 | 31/32 | 32/32 |
| 정확 반복이 있는 작품 | 20/32 | 13/32 | 15/32 |
| 기존 품질 기준 충족 | 2/32 | 8/32 | 3/32 확인 · 1편 미채점 |
| 인과관계 평균 (0~4) | 1.88 | 2.19 | 2.45 |
| 낭독 자연스러움 평균 (0~4) | 2.72 | 3.03 | 2.84 |
| 장면 밀도 평균 (0~4) | 1.66 | 2.41 | 2.19 |

본학습 종료 모델은 원본보다 인과와 완결성, 종료 안정성이 좋아졌습니다. 정책4와 비교하면 낭독과 장면 밀도가 낮아졌고, 첫 출력의 품질 기준 충족 편수도 줄었습니다. 내용 점수는 단일 생성 시드와 같은 API 채점기의 유효 판독 32·32·31편으로 집계했습니다. [9개 내용 항목 전체 집계](reports/independent32.json)

후속 보상안은 형식과 내용 보상을 나눠 훈련 대조를 점검했습니다. 반복과 인과 오류는 감점됐으나 정상 질문도 과하게 감점됐습니다. 사전 기준 8개 중 4개를 충족해 후보 채택을 보류했고, 이 실험의 모델 갱신은 0회입니다. [보상 검증 집계](reports/reward-validation.json)

## 재현 순서

### 1. 환경과 로컬 자료 준비

Windows와 Python 3.11 환경에서 저장소 루트 기준으로 실행합니다. 본학습은 PyTorch 2.7.1과 CUDA 12.8을 사용했습니다. 아래 테스트는 CPU에서 실행됩니다.

```powershell
python -m pip install -r requirements.txt
python -m unittest storymaker.data_augmentation.tests.test_augment storymaker.qwen.tests.test_generation_format
python -m unittest storymaker.qwen.tests.test_evaluation_generate storymaker.qwen.tests.test_evaluation_api storymaker.qwen.tests.test_evaluation_report
python -m unittest storymaker.rlaif.tests.test_reward_pipeline storymaker.rlaif.tests.test_rlaif_objective storymaker.rlaif.tests.test_run_binding
```

| 로컬 경로 | 준비할 자료 |
|---|---|
| `storymaker/data/raw/` · `storymaker/data/processed/v5/` | 사용 권한이 있는 원작과 v5 데이터셋, 토큰화·분할 파일 |
| `storymaker/qwen/models/original/` | 원본 Qwen 가중치와 토크나이저. `STORYMAKER_MODEL_PATH`로 기본 모델 경로 지정 가능 |
| `storymaker/rlaif/rag/train200/` · `storymaker/rlaif/models/bge-m3/` | 훈련 200편으로 고정한 `corpus.jsonl`, `embeddings.npy`, `prepared_manifest.json`과 BGE-M3 가중치 |
| `storymaker/secrets/api_key.yaml` | 유료 API가 필요할 때 로컬에서 `openai_api_key` 값을 설정 |
| `storymaker/qwen/results/` · `storymaker/rlaif/results/` | 보존된 실행 설정, 입력 해시, 체크포인트와 결과 |

원작 수집본과 증강 중간 결과가 있으면 `data_augmentation`의 증강·검수·v5 정리 단계 순서로 사용합니다. 보존된 v5에서 시작할 때는 처리된 데이터와 고정된 원작 단위 분할을 함께 준비합니다. 개발 243편 중 200편은 RLAIF 훈련과 RAG에, 43편은 검증에 사용합니다. 기존 61편은 SFT 진단용이고 독립 32편은 최종 비교용입니다. 과거 실험 파일을 참조하는 테스트에는 해당 실행 기록도 필요합니다.

### 2. LoRA SFT 학습과 고정 테스트 61편 채점

1. 완료한 학습률 비교와 5-fold 결과, 인코딩된 개발 자료, 실행 설정을 준비합니다. 보존된 HPO 실행 폴더에서 이어갈 경우 `config.json`, `folds.json`, `encoded_development.json`, Optuna DB와 보호 해시 파일을 함께 둡니다. 그리드를 새로 구성할 때는 `hpo_sft --prepare-from`으로 완료한 학습률 비교를 불러오며, 보존한 단독 변경 두 결과가 있으면 `--reuse-screen-from`도 지정합니다. 새 실행의 보호 해시를 확인한 뒤 학습합니다.
2. 최종 재학습이 완료된 실행 폴더를 생성·채점·집계에 동일하게 전달합니다. 기본 출력 위치는 실행 폴더 아래 `evaluation/`입니다. 별도 위치를 쓸 때는 세 명령에 같은 `--output-dir`을 전달합니다. 기존 `storymaker/outputs/`의 실행 폴더도 사용할 수 있습니다.
3. 개발 파일럿 생성과 채점을 마친 뒤, 고정 테스트를 생성하고 채점합니다. `evaluation_api`는 기본적으로 계획만 저장합니다. 유료 채점 전에 실행 폴더의 `api_pricing_check.json`에 현재 한국 날짜(`checked_client_date`), 확인한 `model`, `prices_usd_per_million`을 기록하고 아래 설정과 대조합니다. [SFT API 설정](storymaker/qwen/configs/test61_api_settings.json)

```powershell
$sftRun = 'storymaker/qwen/results/sft_hpo_grid'
python -m storymaker.qwen.code.hpo_sft --run-dir $sftRun
python -m storymaker.qwen.code.evaluation_generate --run-dir $sftRun --split pilot
python -m storymaker.qwen.code.evaluation_api --run-dir $sftRun --split pilot
python -m storymaker.qwen.code.evaluation_api --run-dir $sftRun --split pilot --execute
python -m storymaker.qwen.code.evaluation_generate --run-dir $sftRun --split test
python -m storymaker.qwen.code.evaluation_api --run-dir $sftRun --split test
python -m storymaker.qwen.code.evaluation_api --run-dir $sftRun --split test --execute
python -m storymaker.qwen.code.evaluation_report --run-dir $sftRun
```

### 3. RLAIF 배치 갱신과 동일 교훈 검증

1. 훈련 200개 교훈의 순서를 고정해 16편씩 12배치와 마지막 8편으로 나눕니다. 마지막에도 `logical_batch_size`는 16으로 두고 `batch_ids`에 남은 8개만 넣습니다. 각 실행 폴더에 모델·분할·배치 설정(`config.json`), 입력 및 코드 해시(`manifest.json`), 프롬프트, 보상 프로토콜과 준비 검증 결과를 둡니다. 첫 배치는 정책4·가치4에서 시작하고, 이후 배치는 직전에 검증된 정책·가치 및 옵티마이저 상태를 이어받습니다.
2. [활성 실행 경로 예시](storymaker/rlaif/configs/active_run.example.json)를 같은 폴더의 `active_run.json`으로 복사하고, `run_dir`에 지금 실행할 결과 경로를 지정합니다. 설정 파일의 상대 경로는 `storymaker/` 기준입니다. 아래 명령은 배치 편수와 13개 단계의 계획을 저장합니다. `--execute`를 붙이면 생성 → 훈련용 RAG 검색 → API 채점 → 정책·가치 갱신 → 같은 검증 43편 비교로 이어집니다.

```powershell
python -m storymaker.rlaif.code.rlaif_cycle
```

3. 200편 본학습처럼 전체 순회 뒤에 검증할 경우에는 각 배치에서 다음 학습 단계만 차례대로 실행합니다. 유료 단계 전에 날짜별 단가와 공유 비용 장부를 확인합니다. 마지막 배치까지 실제 데이터 ID와 정책·가치 저장 기록을 검증한 뒤, 같은 검증 43편에서 정책4와 마지막 모델을 비교하고 고정 독립 32편으로 최종 진단합니다.

```powershell
python -m storymaker.rlaif.code.rlaif_generate
python -m storymaker.rlaif.code.rlaif_reward_api --prepare-rag
python -m storymaker.rlaif.code.rlaif_reward_api --execute
python -m storymaker.rlaif.code.rlaif_sequence
python -m storymaker.rlaif.code.rlaif_sequence --execute
python -m storymaker.rlaif.code.rlaif_training_report
```

검증 생성에는 `rlaif_evaluate --split validation`을 사용합니다. 원본은 추가 인자 없이, 정책4와 마지막 정책은 각각 `--checkpoint`로 정책 폴더를 지정하고 `--execute`로 생성합니다. 각 정책의 생성이 완료되면 `rlaif_reward_evaluate --policy-tag <태그> --prepare-rag`로 검색 근거를 준비하고, 먼저 계획을 확인한 뒤 같은 태그에 `--execute`를 지정해 채점합니다. 독립 32편은 최종 선택을 기록하고 고정 교훈을 사용해 같은 순서로 진행합니다. 생성은 `--split final`, 채점은 `--phase final`을 지정합니다.

[RLAIF 원 논문](https://arxiv.org/abs/2309.00267) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
