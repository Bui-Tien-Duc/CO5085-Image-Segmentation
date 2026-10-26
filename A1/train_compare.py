"""A1.2 PathMNIST training. No test evaluation or automatic overwrite."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def scores(cm):
    # Rows=true class, columns=predicted class. Includes all 9 classes.
    total = sum(map(sum, cm))
    correct = sum(cm[i][i] for i in range(9))
    f1 = []
    for i in range(9):
        denominator = sum(cm[i]) + sum(row[i] for row in cm)
        f1.append(2 * cm[i][i] / denominator if denominator else 0.0)
    return correct / total if total else 0.0, sum(f1) / 9


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['resnet18', 'vit'], default='resnet18')
    p.add_argument('--mode', choices=['scratch', 'head', 'partial', 'full'], default='head')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parent/'data/pathmnist.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='1 epoch; max 20 train and 10 val batches. Not final results.')
    return p


def main():
    a = args_parser().parse_args()
    if a.epochs < 1 or a.batch_size < 1:
        raise ValueError('epochs and batch-size must be positive')
    if not a.data.is_file():
        raise FileNotFoundError(a.data)
    if a.output.exists():
        raise FileExistsError(f'Choose a new output directory, already exists: {a.output}')
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Select the A1 environment.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda:0')
    pretrained = a.mode != 'scratch'
    mean, std = ((.485,.456,.406),(.229,.224,.225)) if a.model == 'resnet18' else ((.5,.5,.5),(.5,.5,.5))
    if a.model == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 9)
        train_modules = [model.fc] if a.mode == 'head' else [model.layer4,model.fc]
        origin = 'torchvision ResNet18_Weights.IMAGENET1K_V1' if pretrained else 'random initialization'
        extra_version = None
    else:
        import transformers
        from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
        checkpoint = 'google/vit-base-patch16-224'
        processor = ViTImageProcessor.from_pretrained(checkpoint)
        if tuple(processor.image_mean) != mean or tuple(processor.image_std) != std:
            raise ValueError('Checkpoint normalization differs from protocol; review before training.')
        model = ViTForImageClassification.from_pretrained(checkpoint,num_labels=9,ignore_mismatched_sizes=True) if pretrained else ViTForImageClassification(ViTConfig(num_labels=9))
        train_modules = [model.classifier] if a.mode == 'head' else [*model.vit.encoder.layer[-2:],model.vit.layernorm,model.classifier]
        origin = checkpoint if pretrained else 'random ViTConfig initialization (no pretrained weights)'
        extra_version = transformers.__version__
    if a.mode in ('head','partial'):
        for par in model.parameters():
            par.requires_grad = False
        for module in train_modules:
            for par in module.parameters():
                par.requires_grad = True
    # Freeze dropout as well as BatchNorm in frozen modules.
    def train_state():
        if a.mode in ('head','partial'):
            model.eval()
            for module in train_modules:
                module.train()
        else:
            model.train()
    model = model.to(device)
    trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert trainable
    lr = {'scratch':1e-4,'head':1e-3,'partial':1e-5,'full':1e-5}[a.mode]
    optimizer = torch.optim.AdamW([p for _,p in trainable], lr=lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = nn.CrossEntropyLoss()
    class Data(Dataset):
        def __init__(self, split, training):
            with np.load(a.data,allow_pickle=False) as z:
                self.images = z[split+'_images']
                self.labels = z[split+'_labels'].reshape(-1)
            expected = {'train':89996,'val':10004}[split]
            if len(self.images) != expected or len(self.labels) != expected:
                raise ValueError('Unexpected PathMNIST split length')
            if set(np.unique(self.labels)) != set(range(9)):
                raise ValueError('Expected labels 0..8')
            steps = [transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BILINEAR)]
            if training:
                steps.append(transforms.RandomHorizontalFlip(.5))
            self.transform = transforms.Compose(steps+[transforms.ToTensor(),transforms.Normalize(mean,std)])
        def __len__(self):
            return len(self.labels)
        def __getitem__(self,i):
            return self.transform(Image.fromarray(self.images[i]).convert('RGB')),int(self.labels[i])
    # Never load test here.
    generator = torch.Generator().manual_seed(a.seed)
    train = DataLoader(Data('train',True),batch_size=a.batch_size,shuffle=True,num_workers=0,pin_memory=True,generator=generator)
    val = DataLoader(Data('val',False),batch_size=a.batch_size,shuffle=False,num_workers=0,pin_memory=True)
    digest = hashlib.md5()
    with a.data.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != 'a8b06965200029087d5bd730944a56c1':
        raise ValueError('PathMNIST MD5 mismatch; verify dataset before training.')
    config = {'model':a.model,'mode':a.mode,'purpose':'smoke_check_only' if a.smoke else 'full_train_validation',
              'epochs':1 if a.smoke else a.epochs,'batch_size':a.batch_size,'seed':a.seed,'lr':lr,
              'optimizer':'AdamW','weight_decay':.01,'scheduler':'constant','mean':mean,'std':std,
              'resize':'224x224 bilinear','augmentation':'train horizontal flip p=0.5; validation none',
              'pretraining':origin,'selection':'validation macro_f1 highest; tie validation loss lowest',
              'dataset_md5':digest.hexdigest(),'train_size':len(train.dataset),'val_size':len(val.dataset),
              'parameters_total':sum(p.numel() for p in model.parameters()),
              'parameters_trainable':sum(p.numel() for _,p in trainable),'trainable_names':[n for n,_ in trainable],
              'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'transformers':extra_version,
              'python':sys.version,'gpu':torch.cuda.get_device_name(0),'device':'cuda:0',
              'amp':True,'workers':0,'determinism':'requested; unsupported ops warn, not bitwise guaranteed'}
    a.output.mkdir(parents=True)
    def save_json(name, obj):
        (a.output/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    save_json('config.json', config)
    print(json.dumps({k:config[k] for k in ['model','mode','purpose','parameters_trainable','gpu']},indent=2),flush=True)
    best_key = (-math.inf,-math.inf)
    best_epoch = None
    elapsed_train = 0.
    elapsed_val = 0.
    peak = 0.
    fields = ['epoch','train_loss','train_accuracy','train_macro_f1','val_loss','val_accuracy','val_macro_f1','train_seconds','val_seconds','peak_allocated_gib']
    def epoch_run(loader, training, max_batches):
        train_state() if training else model.eval()
        cm = torch.zeros((9,9),dtype=torch.int64,device=device)
        n_seen,total_loss = 0,0.
        for step,(x,y) in enumerate(loader,1):
            if max_batches and step > max_batches:
                break
            x,y = x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with (nullcontext() if training else torch.inference_mode()):
                with torch.autocast('cuda',dtype=torch.float16):
                    logits = model(x) if a.model == 'resnet18' else model(pixel_values=x).logits
                    loss = loss_fn(logits,y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                if training:
                    scaler.scale(loss).backward()
                    if step == 1:
                        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                        assert all(p.grad is not None for _,p in trainable)
                    scaler.step(optimizer)
                    scaler.update()
            total_loss += loss.detach().item()*len(y)
            n_seen += len(y)
            cm += torch.bincount(y*9+logits.detach().argmax(1),minlength=81).reshape(9,9)
            if step == 1 or step % 1000 == 0:
                print(f"{'Train' if training else 'Val'} batch {step}/{min(len(loader),max_batches) if max_batches else len(loader)} loss={loss.item():.4f}",flush=True)
        accuracy,macro = scores(cm.cpu().tolist())
        return total_loss/n_seen,accuracy,macro,n_seen
    with (a.output/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();f.flush()
        for ep in range(1,config['epochs']+1):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            tl,ta,tf,tn=epoch_run(train,True,20 if a.smoke else None)
            torch.cuda.synchronize();ts=time.perf_counter()-t;t=time.perf_counter()
            vl,va,vf,vn=epoch_run(val,False,10 if a.smoke else None)
            torch.cuda.synchronize();vs=time.perf_counter()-t
            mem=torch.cuda.max_memory_allocated()/1024**3;peak=max(peak,mem);elapsed_train+=ts;elapsed_val+=vs
            row=dict(zip(fields,[ep,tl,ta,tf,vl,va,vf,ts,vs,mem]));writer.writerow(row);f.flush()
            key=(vf,-vl)
            if key>best_key:
                best_key=key;best_epoch=ep
                torch.save({'state_dict':model.state_dict(),'config':config,'epoch':ep,'validation_metrics':{'loss':vl,'accuracy':va,'macro_f1':vf}},a.output/'best.pt')
            summary={'status':'in_progress','purpose':config['purpose'],'epochs_completed':ep,'best_epoch':best_epoch,'best_val_macro_f1':best_key[0],'best_val_loss':-best_key[1],
                     'train_seconds':elapsed_train,'validation_seconds':elapsed_val,'peak_allocated_gib':peak,'last_epoch_train_samples':tn,'last_epoch_val_samples':vn,'test_evaluated':False}
            save_json('summary.json',summary)
            print(f'Epoch {ep}/{config["epochs"]} val_accuracy={va:.4f} val_macro_f1={vf:.4f} train_seconds={ts:.1f}',flush=True)
    saved=torch.load(a.output/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['state_dict'],strict=True)
    summary['status']='completed';summary['checkpoint_reload']='ok';save_json('summary.json',summary)
    print('SMOKE CHECK: OK' if a.smoke else 'TRAINING: COMPLETE', a.output,flush=True)


if __name__=='__main__':
    main()
