"""Train-only retrieval and evidence for the direct-RLAIF judge."""
import argparse
import hashlib
import json
import msvcrt
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from storymaker import project as p
from storymaker.rlaif.code.rlaif_environment import OUT, gpu

FOLDER = p.ROOT / 'rlaif/rag/train200'
MODEL = (p.ROOT / 'rlaif/models/bge-m3')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def documents(config):
    manifest = p.read(FOLDER / 'prepared_manifest.json')
    assert sha(FOLDER / 'corpus.jsonl') == manifest['corpus_sha256']
    assert sha(FOLDER / 'embeddings.npy') == manifest['embeddings_sha256']
    docs = [json.loads(line) for line in (FOLDER / 'corpus.jsonl').read_text('utf-8').splitlines()]
    vectors = np.load(FOLDER / 'embeddings.npy', allow_pickle=False)
    train, forbidden = set(config['train_ids']), set(config['validation_ids']) | set(config['excluded_test_ids'])
    sources = {r['source_id'] for r in docs if r['source_id'] is not None}
    assert sources == train and not sources & forbidden
    assert len(docs) == manifest['documents'] and vectors.shape == (len(docs), 1024)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=.002)
    return docs, vectors, manifest


def rank(docs, vectors, query_vector, kind='criterion', k=3):
    scores = vectors @ query_vector
    allowed = [i for i, row in enumerate(docs) if row['kind'] == kind]
    selected = sorted(allowed, key=lambda i: (-float(scores[i]), docs[i]['doc_id']))[:k]
    return [dict(docs[i], similarity=float(scores[i])) for i in selected]


def validate_citations(citations, retrieved):
    available = {row['doc_id']: row for row in retrieved}
    assert isinstance(citations, list) and len(citations) <= 3
    for item in citations:
        assert item['doc_id'] in available
        quote = item['quote']
        assert isinstance(quote, str) and 8 <= len(quote) <= 240
        assert quote in available[item['doc_id']]['text'], 'Evidence quote differs from retrieved source'
    return True


def encoder(device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16 if device == 'cuda' else torch.float32).to(device).eval()
    def encode(texts):
        lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in texts]
        assert max(lengths) <= 2048, 'Split query text explicitly; preserve every chunk'
        batch = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to(device)
        with torch.inference_mode():
            values = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0].float(), p=2, dim=1)
        return values.cpu().numpy(), lengths
    return encode


def validate_index():
    folder = OUT / 'rag_validation'
    folder.mkdir(exist_ok=True)
    assert not (folder / 'verified.json').exists(), 'Inspect existing validation before reuse'
    config_path = p.ROOT / 'rlaif/configs/current.json'
    config = p.read(config_path)
    signatures = {'config': sha(config_path), 'code': sha(__file__), 'corpus': sha(FOLDER / 'corpus.jsonl')}
    docs, vectors, manifest = documents(config)
    books = {r['source_id']: r for r in p.read(p.PROCESSED / 'dataset.json')}
    chosen = [23, 191, 139]
    assert set(chosen) <= set(config['train_ids'])
    queries = []
    for source_id in chosen:
        document = next(r for r in docs if r['source_id'] == source_id)
        renamed = document['text']
        for index, character in enumerate(books[source_id]['output']['characters']):
            renamed = renamed.replace(character['name'], '별이' if index % 2 else '둥이')
        assert renamed != document['text'], 'Choose a document containing character names'
        queries += [(source_id, 'original', document['text']), (source_id, 'renamed', renamed)]
    def status(stage, **extra):
        p.save(folder / 'status.json', {'at_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
            'new_policy_updates': 0, 'paid_API_calls': 0, **extra})
    started = time.monotonic()
    with (p.ROOT / 'rlaif/results/generation.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        observed = gpu()
        assert observed['free_mib'] > 13500 and observed['temperature_c'] < 80
        assert p.read(OUT / 'control.json')['enabled']
        status('loading_retriever')
        encode = encoder('cuda')
        checks = []
        for start in range(0, len(queries), 2):
            values, lengths = encode([q[2] for q in queries[start:start + 2]])
            for (source_id, variant, _), vector, length in zip(queries[start:start + 2], values, lengths):
                hits = rank(docs, vectors, vector, kind='example', k=3)
                checks.append({'expected_source_id': source_id, 'variant': variant, 'query_tokens': length,
                    'hit_at3': source_id in [r['source_id'] for r in hits],
                    'retrieved': [{'doc_id': r['doc_id'], 'source_id': r['source_id'], 'similarity': r['similarity']} for r in hits]})
            status('retrieving', completed=len(checks), total=len(queries))
        assert sha(config_path) == signatures['config'] and sha(__file__) == signatures['code']
        result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'passed': all(r['hit_at3'] for r in checks),
            'checks': checks, 'signatures': signatures, 'RAG_documents': len(docs), 'elapsed_seconds': time.monotonic() - started,
            'scope': 'Six technical original/name-substitution retrieval checks from three training sources; reward validity is a separate test',
            'new_policy_updates': 0, 'paid_API_calls': 0}
        p.save(folder / 'verified.json', result)
        status('passed' if result['passed'] else 'needs_attention', checks=len(checks), hits=sum(r['hit_at3'] for r in checks))
        assert result['passed'], 'Inspect retrieval failures before reward integration'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-index', action='store_true')
    args = parser.parse_args()
    if args.validate_index:
        validate_index()
    else:
        print('Plan: validate train-only retrieval before API reward integration')
