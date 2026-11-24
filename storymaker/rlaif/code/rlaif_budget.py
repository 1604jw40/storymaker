"""One shared RL budget and verified reuse of byte-identical API request content."""
import json
import hashlib
from decimal import Decimal
from pathlib import Path

from storymaker import project as p
def ledger_path(run):
    run=Path(run)
    configured=p.read(run/'config.json').get('shared_budget_ledger')
    if configured is None:
        return run/'budget_ledger.json'
    target=Path(configured).resolve()
    allowed = ((p.ROOT/'rlaif/results').resolve(), (p.ROOT/'outputs').resolve())
    assert any(target.is_relative_to(root) for root in allowed) and target.is_file()
    return target


def lock_path(run):
    return ledger_path(run).with_name('reward_api.lock')


def run_allowance(run, config, ledger):
    cap = config.get('cycle_API_management_cap_usd')
    if cap is None:
        return None
    extension = ledger.get('run_allowances', {}).get(str(Path(run).resolve()))
    if extension is not None:
        assert extension['config_sha256'] == hashlib.sha256((Path(run)/'config.json').read_bytes()).hexdigest()
        assert Decimal(str(extension['previous_cap_usd'])) == Decimal(str(cap))
        cap = extension['cap_usd']
    amount = Decimal(str(cap))
    assert amount.is_finite() and 0 < amount <= Decimal(str(ledger['budget_usd']))
    return amount


def completed_response(record, ledger, body, signature):
    assert record['state']=='completed' and record['request_sha256']==signature
    folder=Path(record['destination'] if record.get('destination') else Path(ledger).parent/record['label']).resolve()
    allowed = [(p.ROOT/'rlaif/results').resolve(), (p.ROOT/'outputs').resolve()]
    config_path = Path(ledger).parent/'config.json'
    if config_path.exists():
        config = p.read(config_path)
        imported = Path(ledger).parent/'budget_import.json'
        if imported.exists():
            receipt = p.read(imported)
            source = Path(config['source_budget_ledger']).resolve()
            assert Path(receipt['source']).resolve() == source
            assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt['source_sha256']
            assert source.parent == Path(config['source_budget_root']).resolve()
            allowed.append(source.parent)
    assert any(folder.is_relative_to(root) for root in allowed)
    request=p.read(folder/'request.json')
    if 'body' in request and 'request_sha256' in request:
        assert request['request_sha256']==signature
        request=request['body']
    assert request==body,'An identical hash must also have identical request content'
    path=folder/'response.json';response=p.read(path)
    assert response['http_status']==200
    raw=json.loads(response['raw_body'])
    assert raw['id']==record['response_id'] and raw['usage']==record['usage']
    assert raw['model']==body['model']
    return path,raw
