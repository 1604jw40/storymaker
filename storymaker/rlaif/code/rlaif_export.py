"""Export the selected, finally evaluated RL policy with standard Hugging Face serialization."""
import argparse
import hashlib
import json
import msvcrt
import re
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

from storymaker import project as p
from storymaker.rlaif.code.rlaif_budget import ledger_path as budget_path, lock_path as budget_lock
from storymaker.rlaif.code.rlaif_environment import OUT, GPU_LOCK
from storymaker.rlaif.code.rlaif_storage import restore_parameters, digest_tensor, save_json
from storymaker.rlaif.code.rlaif_training_report import committed_batch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_selection(root):
    plan = p.read(root/'evaluation/evaluation_plan.json')
    selection_path = root/'evaluation/final_policy_selection.json'
    selected = p.read(selection_path)
    assert selected['selection_used_final_outputs'] is False
    assert selected['final_prompt_sha256'] == plan['final_prompt_sha256'] == sha(plan['final_prompt_file'])
    tag = selected['chosen_tag']
    assert re.fullmatch(r'update_\d{4}', tag)
    batch = committed_batch(root, tag)
    joint = batch/'joint_update_verified.json'
    if not joint.exists():
        joint = batch/'first_update_verified.json'
    committed = p.read(joint)
    checkpoint = Path(committed['policy']).resolve()
    assert checkpoint.name == 'policy' and checkpoint.parent.name == tag
    assert selected['chosen_policy_manifest_sha256'] == sha(checkpoint/'manifest.json')
    training_config = batch/'config.json'
    if not training_config.exists():
        training_config = batch.parent/'config.json'
    for role in ('policy', 'value'):
        receipt = p.read(batch/f'update_{role}_verified.json')
        assert receipt['phase'] == role and receipt['config_sha256'] == sha(training_config)
        assert Path(receipt['checkpoint']).resolve() == Path(committed[role]).resolve()
        assert receipt['trajectories'] == committed['trajectories']
        assert receipt['optimizer_steps'] == 1 and receipt['saved_and_readback_verified']
        assert receipt['changed_tensors'] == p.read(Path(receipt['checkpoint'])/'manifest.json')['changed_tensors'] > 0
    validation_path = root/f'evaluation/reward_metrics/validation_{tag}/summary.json'
    assert selected['validation_summary_sha256'] == sha(validation_path)
    validation = p.read(validation_path)
    final_path = root/f'evaluation/reward_metrics/final_{tag}/summary.json'
    final = p.read(final_path)
    for phase, count, report in [('validation', 43, validation), ('final', 32, final)]:
        assert report['phase'] == phase and report['policy_tag'] == tag and report['prompt_count'] == count
    return checkpoint, {'chosen_tag': tag, 'selection_sha256': sha(selection_path),
        'policy_manifest_sha256': sha(checkpoint/'manifest.json'),
        'validation_summary_sha256': sha(validation_path), 'final_summary_sha256': sha(final_path),
        'final_prompt_sha256': plan['final_prompt_sha256'],
        'committed_batch_receipt_sha256': sha(joint),
        'training_config_path': str(training_config), 'training_config_sha256': sha(training_config)}


def save_and_verify(model, destination):
    assert not torch.cuda.is_initialized(), 'Export is CPU only'
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    assert model.config.tie_word_embeddings
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert not destination.exists(), 'Never overwrite a partial or delivered model'
    destination.mkdir(parents=True)
    # Standard Transformers handles tied weights and safe shard boundaries.
    model.save_pretrained(destination, max_shard_size='2GB')
    state = model.state_dict()
    stored = set()
    files = sorted(destination.glob('*.safetensors'))
    assert files and not list(destination.glob('*.bin'))
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as reader:
            for name in reader.keys():
                assert name in state and name not in stored
                tensor = reader.get_tensor(name)
                assert tensor.dtype == state[name].dtype and tensor.shape == state[name].shape
                assert digest_tensor(tensor) == digest_tensor(state[name]), name
                assert all(torch.isfinite(part).all() for part in tensor.flatten().split(1_000_000)), name
                stored.add(name)
    assert set(state)-stored == {'lm_head.weight'}, 'Only the tied output alias may be omitted'
    return {'serialized_tensors': len(stored), 'tied_output_alias': 'lm_head.weight',
        'weight_sha256': {path.name: sha(path) for path in files},
        'all_saved_weights_readback_equal_and_finite': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    selection_path = OUT/'evaluation/final_policy_selection.json'
    if not args.execute:
        print(json.dumps({'plan_only': True, 'selected_policy_exists': selection_path.exists(),
            'output': str(OUT/'final_model'), 'requirements': 'selected committed policy, validation43 and final32 reports; CPU only; no overwrite'}))
        return
    checkpoint, identity = verify_selection(OUT)
    assert not torch.cuda.is_initialized()
    assert p.read(OUT/'control.json')['enabled']
    assert not (OUT/'final_model').exists(), 'Inspect an existing export; never overwrite it'
    config = p.read(OUT/'config.json')
    source = Path(config['model'])
    source_index = p.read(source/'model.safetensors.index.json')
    source_bytes = sum((source/name).stat().st_size for name in set(source_index['weight_map'].values()))
    assert shutil.disk_usage(OUT).free > source_bytes+20*2**30
    with GPU_LOCK.open('a+b') as lock, (budget_lock(OUT)).open('a+b') as api_lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        api_lock.seek(0)
        msvcrt.locking(api_lock.fileno(), msvcrt.LK_NBLCK, 1)
        assert all(row['state'] == 'completed' for row in p.read(budget_path(OUT))['calls'])
        free_kib = int(subprocess.check_output(['powershell', '-NoProfile', '-NonInteractive', '-Command',
            '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'], text=True, timeout=20).strip())
        assert free_kib*1024 > source_bytes+6*2**30, 'Leave room for CPU master restore and serialization'
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        from storymaker.rlaif.code.rlaif_train import verify_sources
        verify_sources(config, 'export')
        model = AutoModelForImageTextToText.from_pretrained(source, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa')
        restore_parameters(list(model.model.language_model.named_parameters()), checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        settings = p.read(OUT/'evaluation/evaluation_plan.json')['generation']
        for name in ['temperature', 'top_p', 'top_k', 'repetition_penalty', 'max_new_tokens', 'do_sample']:
            setattr(model.generation_config, name, settings[name])
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        destination = OUT/'final_model'
        result = save_and_verify(model, destination)
        tokenizer.save_pretrained(destination)
        for name in ['preprocessor_config.json', 'video_preprocessor_config.json']:
            shutil.copy2(source/name, destination/name)
        restored = AutoTokenizer.from_pretrained(destination, local_files_only=True)
        messages = [{'role': 'system', 'content': (OUT/'system_prompt.txt').read_text('utf-8')},
                    {'role': 'user', 'content': json.dumps({'moral': '작은 도움도 소중해요.'}, ensure_ascii=False)}]
        assert tokenizer.get_vocab() == restored.get_vocab()
        assert tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False) == restored.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        shutil.copy2(OUT/'system_prompt.txt', destination/'system_prompt.txt')
        save_json(destination/'generation_settings.json', settings)
        if config.get('operation_precision'):
            shutil.copy2(p.code_path('rlaif_precision.py'), destination/'rlaif_precision.py')
            save_json(destination/'operation_precision.json',{'mode':config['operation_precision'],
                'stored_weights':'BF16','operations':'FP32','usage':'Load BF16 weights, move language_model to CUDA, leave unused visual on CPU, then rlaif_precision.enable(model, mode) before generate; default BF16 operations differ.'})
        assert sha(identity['training_config_path']) == identity['training_config_sha256']
        for original, name in [(Path(identity['training_config_path']), 'training_config.json'),
                (OUT/'config.json', 'inference_context.json'),
                (OUT/'evaluation/final_policy_selection.json', 'model_selection.json'),
                (OUT/f"evaluation/reward_metrics/validation_{identity['chosen_tag']}/summary.json", 'validation_summary.json'),
                (OUT/f"evaluation/reward_metrics/final_{identity['chosen_tag']}/summary.json", 'final_summary.json'),
                (budget_path(OUT), 'RL_cost_ledger.json')]:
            shutil.copy2(original, destination/name)
        (destination/'MODEL_CARD.md').write_text(
            '# Qwen3.5-4B 그림책 direct-RLAIF 결과\n\n'
            f"선택 정책: {identity['chosen_tag']}. 판단 근거와 실제 결과는 model_selection.json, validation_summary.json, final_summary.json에 있다.\n\n"
            '그림책 SFT 전 원본을 시작점으로 전체 언어 본체를 학습했다. 알고리즘은 REINFORCE + 학습된 가치 기준선 + 고정 참조 KL이며 GRPO가 아니다. '
            '선택 체크포인트의 실제 학습 설정은 training_config.json, 평가·전달 시점 설정은 inference_context.json에 있다. '
            '계산 정밀도는 operation_precision.json을 확인한다. FP32 연산 모드에서는 함께 제공한 rlaif_precision.py를 적용해야 같은 정책이며 기본 BF16 추론은 다를 수 있다. '
            '복구용 FP32 master/optimizer/RNG는 원래 실행 폴더에 보존하며, 이 폴더는 BF16 추론용 표준 Hugging Face 가중치다.\n\n'
            '입력은 system_prompt.txt와 user JSON {"moral": "교훈"}이다. tokenizer.apply_chat_template에서 enable_thinking=False, '
            'add_generation_prompt=True를 사용하고 생성 직전 transformers.set_seed(42)를 적용한다. 디코딩 기본값은 generation_config.json과 '
            'generation_settings.json에 있다. 원본 출력과 EOS/JSON 실패를 그대로 기록한다.\n\n'
            'test61은 이전 진단/출발점 선택에 썼으며 미사용 RL 최종셋이 아니다. 최종32개는 학습/선택에서 분리한 합성 교훈이고 '
            '사람이 검증한 대표 벤치마크가 아니다. API 채점은 전문가 인증이 아니며 같은 API 계열의 제작/보상/평가 편향, 단일 seed, '
            '이미지·사람 검토 미실시 한계가 있다. CE와 자유 생성 품질을 구분한다.\n', encoding='utf-8')
        save_json(destination/'export_verified.json', {**identity, **result,
            'tokenizer_vocabulary_and_generation_prefix_match': True,
            'config_sha256': sha(OUT/'config.json'), 'export_code_sha256': sha(Path(__file__)),
            'artifact_sha256': {path.name: sha(path) for path in destination.iterdir() if path.is_file()},
            'API_calls': 0, 'CUDA_initialized': torch.cuda.is_initialized(),
            'scope': 'CPU serialization and exact weight readback; quality claims belong to preserved final reports'})
        print(json.dumps({'export_verified': str(destination/'export_verified.json'), 'GPU_models_loaded': 0, 'API_calls': 0}))


if __name__ == '__main__':
    main()
