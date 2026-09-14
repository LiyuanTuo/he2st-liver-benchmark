"""Training and inference. Validation chooses checkpoints; fit never opens test RNA."""
from contextlib import nullcontext
from pathlib import Path
import copy
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SlideDataset, gene_digest, read_panel, training_neighbor_targets
from .metrics import per_gene_pearson
from .models import ContextFusion


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    temp.replace(path)


def scores(true, predicted, mask):
    r = per_gene_pearson(true, predicted)
    def mean(a):
        return float(np.nanmean(a)) if np.isfinite(a).any() else None
    return {'HEG200': mean(r), 'HEG50': mean(r[mask]),
            'MAE': float(np.abs(true-predicted).mean()), 'defined_genes': int(np.isfinite(r).sum())}


def validate_config(config):
    groups = [set(config[key]) for key in ['train', 'validation', 'test']]
    if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Train, validation and test slides must be nonempty and disjoint')
    if config['epochs'] <= 0 or config['patience'] <= 0 or config['batch_size'] <= 0:
        raise ValueError('Epochs, patience and batch size must be positive')
    if config['backbone_lr'] <= 0 or not 0 <= config['ema_decay'] < 1:
        raise ValueError('Invalid learning rate or EMA decay')


def environment(device):
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = False
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use a global CUDA Python or --device cpu')
    return {'python_torch': torch.__version__, 'device': device,
            'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None}


def prepare(images, device, augment=False, color_strength=.08):
    images = images.to(device, non_blocking=True).float().div_(255)
    if augment:
        images = torch.rot90(images, random.randrange(4), [-2, -1])
        if random.random() < .5:
            images = images.flip(-1)
        # Per-spot color variation shared by that spot's views; preserves geometry.
        gain = 1 + (torch.rand(len(images), 1, 3, 1, 1, device=device)*2-1)*color_strength
        shift = (torch.rand(len(images), 1, 3, 1, 1, device=device)*2-1)*.025
        images = (images*gain+shift).clamp_(0, 1)
    center = images.new_tensor([.485, .456, .406])[None, None, :, None, None]
    scale = images.new_tensor([.229, .224, .225])[None, None, :, None, None]
    return (images-center)/scale


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()


@torch.inference_mode()
def infer(model, loader, mean, std, device, tta=False):
    model.eval()
    rows = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = prepare(images, device)
        with autocast(device):
            if tta:
                output = torch.stack([model(torch.rot90(images, k, [-2, -1])) for k in range(4)]).float().mean(0)
            else:
                output = model(images).float()
        rows.append((output*std+mean).cpu().numpy())
    return np.concatenate(rows)


def build_dataset(root, config, slides, genes, labels=True):
    return SlideDataset(root, slides, config['arrays'], config['images'], config['scales'], genes, labels)


def fit(root, config, output, device='cuda'):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (output/'selection.json').exists():
        saved = json.loads((output/'selection.json').read_text())
        if saved['config_sha256'] != config_hash:
            raise ValueError('Output already belongs to a different config; use a new output directory')
        print(f'Reusing completed fit: {output}', flush=True)
        return saved
    if (output/'history.json').exists():
        raise RuntimeError('Interrupted fit found. Preserve it and choose a new output directory; do not silently overwrite.')
    env = environment(device)
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    genes, mask = read_panel(Path(root)/config['panel'])
    train = build_dataset(root, config, config['train'], genes)
    validation = build_dataset(root, config, config['validation'], genes)
    raw_train = train.y.copy()
    train.y = training_neighbor_targets(train, config['label_smoothing'])
    workers = config.get('workers', 2)
    loader_options = dict(num_workers=workers, pin_memory=device=='cuda', persistent_workers=workers>0)
    train_loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, **loader_options)
    validation_loader = DataLoader(validation, batch_size=config['batch_size'], **loader_options)
    mean = torch.tensor(raw_train.mean(0), device=device)
    std = torch.tensor(np.maximum(raw_train.std(0), .1), device=device)
    architecture = dict(backbone=config['backbone'], scales=len(config['scales']), genes=len(genes), hidden=config['hidden'])
    model = ContextFusion(**architecture, pretrained=config['pretrained']).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['backbone_lr']},
        {'params': model.head.parameters(), 'lr': config['head_lr']},
    ], weight_decay=config['weight_decay'])
    def schedule(epoch):
        warmup = config.get('warmup_epochs', 2)
        if epoch < warmup:
            return (epoch+1)/max(warmup, 1)
        phase = (epoch-warmup)/max(config['epochs']-warmup, 1)
        return .1 + .9*.5*(1+math.cos(math.pi*phase))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    protocol = {'config': config, 'config_sha256': config_hash, 'genes_sha256': gene_digest(genes),
                'environment': env, 'selection': 'C1 HEG200 PCC, raw/EMA checkpoint then 1/4-view TTA',
                'test_RNA_read_during_fit': False, 'train_spots': len(train), 'validation_spots': len(validation)}
    write_json(output/'protocol.json', protocol)
    history, best, stale, updates = [], -float('inf'), 0, 0
    start = time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        model.train()
        losses = []
        for images, labels in train_loader:
            target = (labels.to(device)-mean)/std
            with autocast(device):
                predicted = model(prepare(images, device, True, config['color_strength'])).float()
            pc = predicted-predicted.mean(0)
            tc = target-target.mean(0)
            correlation = (pc*tc).sum(0)/(pc.square().sum(0)*tc.square().sum(0)+1e-6).sqrt()
            loss = (predicted-target).square().mean() + config['correlation_loss']*(1-correlation.mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            updates += 1
            decay = min(config['ema_decay'], (1+updates)/(10+updates))
            with torch.no_grad():
                current = model.state_dict()
                for key, value in ema.state_dict().items():
                    source = current[key]
                    if value.is_floating_point():
                        value.lerp_(source, 1-decay)
                    else:
                        value.copy_(source)
            losses.append(loss.item())
        candidates = {}
        improved = False
        for kind, candidate in [('raw', model), ('ema', ema)]:
            pred = infer(candidate, validation_loader, mean, std, device)
            metric = scores(validation.y, pred, mask)
            candidates[kind] = metric
            if metric['defined_genes'] != len(genes):
                continue
            if metric['HEG200'] > best + 1e-5:
                best = metric['HEG200']; improved = True
                checkpoint = {'model': candidate.state_dict(), 'architecture': architecture,
                    'mean': mean.cpu(), 'std': std.cpu(), 'genes': genes.tolist(), 'heg50_mask': mask.tolist(),
                    'epoch': epoch, 'kind': kind, 'config': config, 'config_sha256': config_hash}
                temp = output/'best.tmp.pt'; torch.save(checkpoint, temp); temp.replace(output/'best.pt')
        stale = 0 if improved else stale+1
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'validation': candidates,
                        'elapsed_seconds': time.perf_counter()-start, 'backbone_lr': optimizer.param_groups[0]['lr']})
        write_json(output/'history.json', history)
        print(json.dumps(history[-1]), flush=True)
        scheduler.step()
        if stale >= config['patience']:
            break
    best_checkpoint = torch.load(output/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model'])
    candidates = {str(tta): scores(validation.y, infer(model, validation_loader, mean, std, device, tta), mask) for tta in [False, True]}
    tta = max([False, True], key=lambda t: candidates[str(t)]['HEG200'])
    selection = {'config_sha256': config_hash, 'epoch': best_checkpoint['epoch'], 'weights': best_checkpoint['kind'],
                 'tta': tta, 'validation': candidates[str(tta)], 'tta_candidates': candidates,
                 'locked_unix': time.time(), 'test_RNA_read': False}
    write_json(output/'selection.json', selection)
    pred = infer(model, validation_loader, mean, std, device, tta)
    np.savez_compressed(output/'validation.npz', predicted=pred, truth=validation.y, genes=genes,
        barcodes=np.concatenate([m['barcodes'] for m in validation.metadata]),
        coordinates_xy=np.concatenate(validation.xy), heg50_mask=mask)
    print('FIT COMPLETE', selection, flush=True)
    return selection


def predict(root, checkpoint_path, slides, output, device='cuda'):
    environment(device)
    checkpoint_path = Path(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    selection = json.loads(checkpoint_path.with_name('selection.json').read_text())
    if saved['config_sha256'] != selection['config_sha256']:
        raise ValueError('Checkpoint/selection configuration mismatch')
    model = ContextFusion(**saved['architecture']).to(device)
    model.load_state_dict(saved['model'])
    data = build_dataset(root, saved['config'], slides, saved['genes'], labels=False)
    loader = DataLoader(data, batch_size=saved['config']['batch_size'],
                        num_workers=saved['config'].get('workers', 2), pin_memory=device=='cuda')
    values = infer(model, loader, saved['mean'].to(device), saved['std'].to(device), device, selection['tta'])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, predicted=values, genes=np.asarray(saved['genes']), heg50_mask=saved['heg50_mask'],
        barcodes=np.concatenate([m['barcodes'] for m in data.metadata]), coordinates_xy=np.concatenate(data.xy))
    return output
