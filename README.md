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

## 로컬 검증

Windows와 Python 3.11 환경에서 아래 명령을 저장소 루트에서 실행합니다. 이전 실행은 PyTorch 2.7.1과 CUDA 12.8을 사용했습니다. 아래 테스트는 CPU에서 실행됩니다.

    python -m pip install -r requirements.txt
    python -m unittest storymaker.data_augmentation.tests.test_augment storymaker.qwen.tests.test_generation_format storymaker.rlaif.tests.test_reward_pipeline storymaker.rlaif.tests.test_rlaif_objective

실험 재현에는 사용 권한이 있는 원작·v5 데이터, Qwen 가중치, 훈련 RAG 인덱스, 해당 실행의 설정과 체크포인트가 필요합니다. 데이터는 **storymaker/data/**, Qwen 기본 가중치는 **storymaker/qwen/models/original/**, 인덱스는 **storymaker/rlaif/rag/train200/**, BGE-M3 가중치는 **storymaker/rlaif/models/bge-m3/**에서 로컬로 관리합니다. **STORYMAKER_MODEL_PATH**로 Qwen 경로를 지정할 수 있습니다. 유료 API의 키는 Git에서 제외하는 **storymaker/secrets/api_key.yaml**에 로컬로 둡니다. 과거 실행을 대조하는 일부 테스트에는 해당 실행 기록도 필요합니다.

데이터는 훈련200·검증43·이전 SFT 진단61·독립 테스트32 역할로 구분했습니다. 완성 모델과 생성·채점 원문은 프로젝트 실행 저장소에서 관리합니다.

[RLAIF 원 논문](https://arxiv.org/abs/2309.00267) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
