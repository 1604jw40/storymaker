"""완료된 SFT와 원본의 동일 조건 생성. API 호출·학습·실패 출력 재추출은 하지 않는다."""
import argparse
import hashlib
import json
import msvcrt
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from storymaker.qwen.code import generate_story as generation
from storymaker import project as p
from storymaker.qwen.code import watch_sft as w
def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def completed_context(run):
    if p.read(run / "status.json")["state"] != "complete":
        raise ValueError("Final SFT has not completed")
    if w.lock_held(run / "run.lock") or w.lock_held(run / "watchdog/watchdog.lock") or w.processes(run):
        raise ValueError("Training or its watchdog still owns the run")
    config = p.read(run / "config.json")
    selected, final = p.read(run / "selection.json"), p.read(run / "final_result.json")
    books = p.read(p.PROCESSED / "dataset.json")
    encoded = p.read(run / "encoded_development.json")
    dev = {r["source_id"] for r in books if r["source_split"] != "test"}
    test = {r["source_id"] for r in books if r["source_split"] == "test"}
    assert len(books) == len({r["source_id"] for r in books}) == 304
    p.check_folds(p.read(run / "folds.json"), dev, test)
    assert len(encoded) == 243 and {r["source_id"] for r in encoded} == dev
    fit = final["fit"]
    assert final["selection"] == selected and not final["test_used_for_training"] and not final["test_used_for_selection"]
    assert not selected["test_used_for_selection"] and selected["heldout_test_count"] == 61
    assert fit["source_ids"] == sorted(dev) and fit["train_count"] == 243 and fit["validation_count"] == 0
    assert fit["epochs"] == selected["epoch"] and fit["hyperparameters"] == selected["hyperparameters"]
    assert fit["learning_rate"] == selected["learning_rate"] == config["fixed_learning_rate"]
    assert fit["run_config_sha256"] == digest(run / "config.json") and not fit["test_used_for_training"]
    assert final["folds_sha256"] == digest(run / "folds.json")
    assert final["encoded_development_sha256"] == digest(run / "encoded_development.json")
    for name, expected in config["dataset_sha256"].items():
        assert digest(p.PROCESSED / name) == expected
    assert final["dataset_sha256"] == config["dataset_sha256"]
    assert config["dtype"] == "bfloat16" and config["quantization"] is False and config["seed"] == 42
    adapter = Path(fit["adapter"]).resolve()
    assert adapter == (run / "final_refit/adapter").resolve()
    assert (adapter / "adapter_model.safetensors").is_file() and (adapter / "adapter_config.json").is_file()
    model = Path(config["model"])
    weights = list(model.glob("*.safetensors")) + list(model.glob("pytorch_model*.bin"))
    assert weights, "Original model weights missing"
    files = [run / n for n in ("config.json", "selection.json", "final_result.json", "folds.json", "encoded_development.json")]
    files += [p.PROCESSED / n for n in config["dataset_sha256"]]
    files += [p.ROOT / "data/raw/aesop.json", p.ROOT / "data/source/system_prompt.txt",
              Path(generation.__file__), Path(__file__), adapter / "adapter_model.safetensors", adapter / "adapter_config.json"]
    files += weights + [x for x in model.iterdir() if x.is_file() and
                        (x.name.startswith(("tokenizer", "chat_template", "special_tokens")) or x.name in {"config.json", "generation_config.json", "vocab.json", "merges.txt", "model.safetensors.index.json"})]
    return {"config": config, "books": books, "encoded": encoded, "adapter": adapter, "files": files}


def manifest_for(context):
    encoded = sorted(context["encoded"], key=lambda r: (len(r["input_ids"]), r["source_id"]))
    pilot = [encoded[i]["source_id"] for i in (0, len(encoded) // 2, len(encoded) - 1)]
    test = sorted(r["source_id"] for r in context["books"] if r["source_split"] == "test")
    assert len(set(pilot)) == 3 and len(test) == 61 and not set(pilot) & set(test)
    identity = {"version": "paired-generation-v1", "source_sha256": {str(path): digest(path) for path in sorted(set(context["files"]))},
                "decoding": generation.DECODING, "seed": 42, "pilot_ids": pilot, "test_ids": test,
                "pilot_selection": "development shortest/median/longest encoded examples; technical checks only",
                "model_variants": ["base", "sft"], "model_revision": context["config"]["model_revision"]}
    return {"identity": identity, "sha256": signature(identity)}


def require_frozen_protocol(output, manifest):
    frozen = p.read(output / "protocol_frozen.json")
    assert frozen["generation_manifest_sha256"] == manifest["sha256"]
    for key in ("pilot_api_report", "api_settings", "rubric"):
        assert digest(frozen[key]) == frozen[key + "_sha256"]
    report = p.read(frozen["pilot_api_report"])
    assert report["phase"] == "pilot" and report["completed_pairs"] == 3 and report["all_responses_complete"]


def jobs_for(run, output, context, manifest, split):
    if split not in {"pilot", "test"}:
        raise ValueError("Unknown evaluation split")
    rows = {r["source_id"]: r for r in context["books"]}
    jobs = []
    for index, number in enumerate(manifest["identity"][split + "_ids"]):
        for variant in (("base", "sft") if index % 2 == 0 else ("sft", "base")):
            jobs.append({"run_dir": str(run), "output_dir": str(output), "split": split, "source_id": number,
                         "variant": variant, "moral": rows[number]["input"]["moral"],
                         "manifest_sha256": manifest["sha256"],
                         "output": str(output / "generations" / split / str(number) / (variant + ".json"))})
    return jobs


def worker(run, job_path):
    context = completed_context(run)
    job = p.read(job_path)
    output = Path(job["output_dir"]).resolve()
    assert (p.ROOT / "outputs").resolve() in output.parents
    manifest = p.read(output / "generation_manifest.json")
    assert manifest["sha256"] == signature(manifest["identity"])
    assert job in jobs_for(run, output, context, manifest, job["split"])
    if job["split"] == "test":
        require_frozen_protocol(output, manifest)
    assert not Path(job["output"]).exists(), "Never overwrite a generated output"
    p.RUN = run
    started = time.monotonic()
    result = generation.generate(job["moral"], context["adapter"] if job["variant"] == "sft" else None)
    validate_generation(result, job, context)
    p.save(job["output"], {"state": "completed", "job": job, "seconds": time.monotonic() - started, "generation": result})


def validate_generation(result, job, context):
    expected = {"moral": job["moral"], "model_variant": job["variant"],
                "adapter": str(context["adapter"]) if job["variant"] == "sft" else None,
                "decoding": generation.DECODING, "seed": 42,
                "base_model_revision": context["config"]["model_revision"]}
    assert all(result.get(k) == v for k, v in expected.items()), "Generation settings mismatch"
    assert isinstance(result["raw_text"], str) and isinstance(result["format_problems"], list)


def batch(run, output, split):
    context = completed_context(run)
    manifest = manifest_for(context)
    path = output / "generation_manifest.json"
    if path.exists():
        assert p.read(path) == manifest, "Evaluation inputs changed; do not mix outputs"
    else:
        p.save(path, manifest)
    if split == "test":
        require_frozen_protocol(output, manifest)
    jobs = jobs_for(run, output, context, manifest, split)
    completed = 0
    for job in jobs:
        result_path = Path(job["output"])
        failure = result_path.with_suffix(".failure.json")
        if failure.exists():
            raise RuntimeError(f"Unresolved technical generation failure: {failure}")
        if result_path.exists():
            cached = p.read(result_path)
            assert cached["state"] == "completed" and cached["job"] == job, "Cached output belongs to another job"
        else:
            job_path = output / "jobs" / f"{split}_{job['source_id']}_{job['variant']}.json"
            p.save(job_path, job)
            p.save(output / "generation_status.json", {"state": "generating", "split": split,
                   "completed_outputs": completed, "total_outputs": len(jobs), "current_job": job,
                   "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.with_suffix(".stdout.log").open("a", encoding="utf-8") as out, result_path.with_suffix(".stderr.log").open("a", encoding="utf-8") as err:
                result = subprocess.run([context["config"]["python"], "-X", "utf8", "-u", str(Path(__file__).resolve()),
                          "--run-dir", str(run), "--worker-job", str(job_path)], stdout=out, stderr=err,
                          creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode or not result_path.exists():
                p.save(failure, {"job": job, "returncode": result.returncode, "automatic_retry": False})
                raise RuntimeError(f"Generation worker failed; inspect {failure}")
            cached = p.read(result_path)
            assert cached["state"] == "completed" and cached["job"] == job
        validate_generation(cached["generation"], job, context)
        # JSON/내용 실패도 완료된 시도로 남겨 다시 뽑지 않는다.
        completed += 1
    assert manifest_for(context) == manifest, "Source files changed during generation"
    p.save(output / "generation_status.json", {"state": "phase_complete", "split": split,
           "completed_outputs": completed, "total_outputs": len(jobs), "api_evaluation": "not_run",
           "updated_at_utc": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--worker-job", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = (args.output_dir or run / "evaluation").resolve()
    if any((p.ROOT / "outputs").resolve() not in path.parents for path in (run, output)):
        raise ValueError("Evaluation must stay inside project outputs")
    if args.worker_job:
        worker(run, args.worker_job)
        return
    with (p.ROOT / "outputs/generation.lock").open("a+b") as lock:
        if not lock.tell():
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        batch(run, output, args.split)


if __name__ == "__main__":
    main()
