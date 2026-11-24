"""Bounded-memory full-parameter gradient accumulation and immutable checkpoints."""
import hashlib
import math
import json
import time
from pathlib import Path

import numpy as np
import torch
from storymaker import project as p
def save_json(path, value):
    for attempt in range(6):
        try:
            return p.save(path, value)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)


def digest_tensor(tensor):
    digest = hashlib.sha256()
    for chunk in tensor.detach().flatten().split(1_000_000):
        digest.update(chunk.to('cpu').contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class GradientBuffer:
    """Mutable FP32 scratch with one parameter mapped at a time, never a model checkpoint."""
    def __init__(self, folder, parameters):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.parameters=list(parameters);self.spans={};offset=0
        for name,value in self.parameters:
            assert name not in self.spans
            self.spans[name]=(offset,offset+value.numel(),tuple(value.shape));offset+=value.numel()
        self.target=self.folder/'gradient_fp32.bin'
        assert not self.target.exists(),'Inspect an existing working buffer before reuse'
        with self.target.open('xb') as stream:
            stream.truncate(offset*4)
        self.maps={};self.handles=[];self.seen=set();self.peak_mapped_bytes=0
        save_json(self.folder/'manifest.json', {'scope':'mutable FP32 gradient scratch, not a checkpoint',
            'elements':offset,'spans':self.spans,'state':'accumulating','mapping':'one parameter window'})

    def view(self,name):
        assert not self.maps or name in self.maps,'Release previous parameter before mapping another'
        if name not in self.maps:
            start,stop,shape=self.spans[name]
            self.maps[name]=np.memmap(self.target,dtype=np.float32,mode='r+',offset=start*4,shape=shape)
            self.peak_mapped_bytes=max(self.peak_mapped_bytes,(stop-start)*4)
        return torch.from_numpy(self.maps[name])

    def release(self,name):
        # Call only after tensor views/optimizer.grad referencing this window have been dropped.
        array=self.maps.pop(name)
        array.flush();array._mmap.close()

    def install(self):
        def hook_for(name):
            def hook(parameter):
                grad=parameter.grad.detach().flatten();accumulator=self.view(name).flatten()
                for start in range(0,grad.numel(),1_000_000):
                    accumulator[start:start+1_000_000].add_(grad[start:start+1_000_000].to('cpu',dtype=torch.float32))
                parameter.grad=None
                del accumulator
                self.release(name);self.seen.add(name)
            return hook
        self.handles=[value.register_post_accumulate_grad_hook(hook_for(name)) for name,value in self.parameters]

    def finish(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
        assert not self.maps and self.seen==set(self.spans)
        norm2=0.
        for name in self.spans:
            flat=self.view(name).flatten()
            for start in range(0,len(flat),1_000_000):
                part=flat[start:start+1_000_000]
                assert torch.isfinite(part).all(),name
                norm2+=float(part.double().square().sum())
                del part
            del flat
            self.release(name)
        manifest=p.read(self.folder/'manifest.json')
        manifest.update(state='complete_gradient',peak_mapped_bytes=self.peak_mapped_bytes)
        save_json(self.folder/'manifest.json',manifest)
        return norm2**.5

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
        for name in list(self.maps):self.release(name)


def save_parameter(folder, name, master, optimizer, bf16_hash):
    folder = Path(folder)
    stem = hashlib.sha256(name.encode()).hexdigest()
    target = folder / (stem + '.pt')
    assert not target.exists(), 'Never overwrite a committed parameter state'
    chunks = []
    for index, chunk in enumerate(master.detach().flatten().split(16_000_000)):
        path = folder / (stem + '_' + str(index) + '.pt')
        assert not path.exists()
        # clone prevents a small view serializing the entire >2GB underlying storage.
        torch.save(chunk.clone(), path)
        chunks.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    torch.save({'name': name, 'shape': list(master.shape), 'chunks': chunks,
                'optimizer': optimizer.state_dict(), 'bf16_sha256': bf16_hash}, target)
    return {'name': name, 'file': target.name,
            'metadata_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'bf16_sha256': bf16_hash}


def read_parameter(folder, record):
    folder = Path(folder)
    path = folder / record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['metadata_sha256']
    metadata = torch.load(path, map_location='cpu', weights_only=True)
    assert metadata['name'] == record['name']
    master = torch.empty(metadata['shape'], dtype=torch.float32)
    offset = 0
    for item in metadata['chunks']:
        path = folder / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        part = torch.load(path, map_location='cpu', weights_only=True)
        master.flatten()[offset:offset+part.numel()].copy_(part)
        offset += part.numel()
    assert offset == master.numel()
    return master, metadata['optimizer']


def optimizer_step(parameters, gradients, destination, learning_rate, previous=None, status=None):
    assert type(learning_rate) in (int, float) and math.isfinite(learning_rate) and learning_rate > 0
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / 'manifest.json').exists()
    previous_records = {}
    if previous is not None:
        previous_records = {x['name']: x for x in p.read(Path(previous) / 'manifest.json')['parameters']}
    records = []
    changed = 0
    for index, (name, parameter) in enumerate(sorted(parameters, key=lambda item: item[1].numel(), reverse=True)):
        before = digest_tensor(parameter)
        old_state = None
        if previous is None:
            master = torch.nn.Parameter(parameter.detach().to('cpu', dtype=torch.float32))
        else:
            value, old_state = read_parameter(previous, previous_records[name])
            master = torch.nn.Parameter(value)
        master.grad = gradients.view(name)
        optimizer = torch.optim.Adafactor([master], lr=learning_rate, weight_decay=0., foreach=False)
        if old_state is not None:
            optimizer.load_state_dict(old_state)
        # Loading optimizer state restores its old hyperparameters as well as moments.
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        gradients.release(name)
        assert torch.isfinite(master).all(), name
        with torch.no_grad():
            parameter.copy_(master)
        after = digest_tensor(parameter)
        changed += after != before
        records.append(save_parameter(destination, name, master, optimizer, after))
        del master, optimizer, old_state
        if status is not None and index % 40 == 0:
            status('optimizer', completed_parameters=index+1, total_parameters=len(parameters))
    assert changed > 0
    torch.save({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if any(x.is_cuda for _, x in parameters) else []},
               destination / 'rng.pt')
    manifest = {'parameters': records, 'changed_tensors': changed, 'optimizer': 'FP32 CPU torch Adafactor',
                'previous': str(previous) if previous else None, 'learning_rate': learning_rate,
                'rng_sha256': hashlib.sha256((destination / 'rng.pt').read_bytes()).hexdigest()}
    # Validate actual readback before publishing the committed manifest.
    for record in records:
        master, state = read_parameter(destination, record)
        assert digest_tensor(master.to(dict(parameters)[record['name']].dtype)) == record['bf16_sha256']
        assert int(state['state'][0]['step']) >= 1
        del master, state
    save_json(destination / 'manifest.json', manifest)
    return manifest


def restore_parameters(parameters, source):
    records = {x['name']: x for x in p.read(Path(source) / 'manifest.json')['parameters']}
    assert set(records) == {name for name, _ in parameters}
    for name, parameter in parameters:
        master, _ = read_parameter(source, records[name])
        with torch.no_grad():
            parameter.copy_(master)
        assert digest_tensor(parameter) == records[name]['bf16_sha256']
        del master
