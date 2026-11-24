"""공통 경로·입출력·분할 검증. 전처리와 EDA의 실행 코드는 notebooks/에 둔다."""

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "qwen/results/sft_hpo_grid"
PROCESSED = ROOT / "data/processed/v5"
DATA = PROCESSED
MODEL = Path(
    os.environ.get(
        "STORYMAKER_MODEL_PATH", str(ROOT / "qwen/models/original")
    )
)
PYTHON = Path(sys.executable)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def check_folds(folds, dev_ids, test_ids):
    """같은 원작은 train/validation/test를 넘나들지 않는다."""
    assert len(folds) == 5 and len(dev_ids) == 243 and len(test_ids) == 61
    assert not dev_ids & test_ids
    seen = []
    for fold in folds:
        train, val = set(fold["train_ids"]), set(fold["validation_ids"])
        assert len(train) == len(fold["train_ids"]) and len(val) == len(
            fold["validation_ids"]
        )
        assert not train & val and train | val == dev_ids
        assert not (train | val) & test_ids
        seen.extend(fold["validation_ids"])
    assert len(seen) == len(set(seen)) == 243 and set(seen) == dev_ids


def code_path(name):
    """Resolve a known worker basename across the three project phases."""
    assert Path(name).name == name and name.endswith('.py')
    candidates = [ROOT / phase / 'code' / name for phase in ('rlaif', 'qwen', 'data_augmentation')]
    candidates.append(ROOT / name)
    matches = [path for path in candidates if path.is_file()]
    assert len(matches) == 1, (name, matches)
    return matches[0]


def worker_command(name, *arguments, python=None):
    module = '.'.join(code_path(name).relative_to(ROOT.parent).with_suffix('').parts)
    return [str(python or PYTHON), '-X', 'utf8', '-m', module, *map(str, arguments)]
