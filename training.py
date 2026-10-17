"""Seeded supervised training, validation-selected checkpoints and held-out evaluation.

The repository-specific adapter supplies build_model(config). NPZ inputs contain
x_train/y_train, x_val/y_val and x_test/y_test. Test labels never select a checkpoint.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, pred, classes):
    cm=np.zeros((classes,classes),dtype=int)
    np.add.at(cm,(y,pred),1)
    tp=cm.diagonal();precision=np.divide(tp,cm.sum(0),out=np.zeros(classes,dtype=float),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(classes,dtype=float),where=cm.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(classes,dtype=float),where=precision+recall>0)
    return {'rows':len(y),'accuracy':float((y==pred).mean()),'macro_f1':float(f1.mean()),'confusion_matrix':cm.tolist(),'class_recall':recall.tolist()}


def main(build_model, default_config, binary=False, tokens=False):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--seed',type=int,default=7);ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--config',type=Path);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--evaluate-only',action='store_true');ap.add_argument('--resume',action='store_true')
    ap.add_argument('--precision-floor',type=float,default=.72)
    a=ap.parse_args()
    if a.epochs<1 or a.batch_size<2 or a.lr<=0:raise ValueError('Positive epochs/lr and batch size >=2 required')
    if a.evaluate_only and a.checkpoint is None:raise ValueError('--evaluate-only requires --checkpoint')
    if a.resume and a.checkpoint is None:raise ValueError('--resume requires --checkpoint')
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Use a new empty output directory; prior results are never overwritten')
    a.output.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config=dict(default_config)
    if a.config:config.update(json.loads(a.config.read_text()))
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True) if a.checkpoint else None
    if ck:
        config=ck['config'];a.seed=int(ck['seed'])
    model=build_model(config).cpu();optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    best=float('inf');start=0;history=[]
    rng=torch.Generator().manual_seed(a.seed)
    if ck:
        model.load_state_dict(ck['model_state'],strict=True)
        if a.resume:
            if ck['data_sha256']!=sha(a.data):raise ValueError('Resume requires identical prepared dataset bytes')
            optimizer.load_state_dict(ck['optimizer_state']);start=ck['epoch'];best=ck['validation_loss'];history=ck['history']
            if best>min(r['validation_loss'] for r in history)+1e-10:raise ValueError('Resume from the selected best.pt checkpoint, not last.pt')
            torch.set_rng_state(ck['torch_rng']);rng.set_state(ck['loader_rng'])
    with np.load(a.data,allow_pickle=False) as z:
        arrays={key:z[key].copy() for key in ('x_train','y_train','x_val','y_val','x_test','y_test')}
    classes=2 if binary else int(config['classes'])
    loaders={}
    for split in ('train','val','test'):
        x,y=arrays['x_'+split],arrays['y_'+split]
        if len(x)!=len(y) or len(y)<2 or y.ndim!=1 or not np.isfinite(x).all() or not np.isin(y,range(classes)).all():raise ValueError('Invalid aligned finite dataset or class labels: '+split)
        if tokens and (not np.issubdtype(x.dtype,np.integer) or x.min()<0 or x.max()>=config['vocabulary']):raise ValueError('Token IDs must be integers inside the training vocabulary')
        tx=torch.from_numpy(x).long() if tokens else torch.from_numpy(x).float()
        if x.dtype==np.uint8 and x.ndim==4:tx=tx.div(127.5).sub(1.)
        loaders[split]=DataLoader(TensorDataset(tx,torch.from_numpy(y).long()),batch_size=a.batch_size,shuffle=split=='train',generator=rng if split=='train' else None)
    loss_fn=torch.nn.BCEWithLogitsLoss() if binary else torch.nn.CrossEntropyLoss()
    def evaluate(split):
        model.eval();total=0.;ys=[];probs=[]
        with torch.no_grad():
            for x,y in loaders[split]:
                logits=model(x);target=y.float().reshape(-1,1) if binary else y
                loss=loss_fn(logits,target);total+=loss.item()*len(y)
                ys.extend(y.tolist());probs.extend(logits.sigmoid().reshape(-1).tolist() if binary else logits.softmax(-1).tolist())
        return total/len(ys),np.array(ys),np.array(probs)
    if not a.evaluate_only:
        if start>=a.epochs:raise ValueError('Resume total epochs must exceed checkpoint epoch')
        for epoch in range(start,a.epochs):
            model.train();total=0.;n=0
            for x,y in loaders['train']:
                optimizer.zero_grad(set_to_none=True);logits=model(x)
                loss=loss_fn(logits,y.float().reshape(-1,1) if binary else y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=loss.item()*len(y);n+=len(y)
            vl,_,_=evaluate('val');history.append({'epoch':epoch+1,'train_loss':total/n,'validation_loss':vl})
            print(json.dumps(history[-1]),flush=True)
            state={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'config':config,'epoch':epoch+1,'seed':a.seed,'validation_loss':vl,'history':history,'data_sha256':sha(a.data),'torch_rng':torch.get_rng_state(),'loader_rng':rng.get_state()}
            torch.save(state,a.output/'last.pt')
            if vl<best:best=vl;torch.save(state,a.output/'best.pt')
        # If resuming and no later epoch improves, the supplied checkpoint remains the selected one.
        selected=a.output/'best.pt' if (a.output/'best.pt').exists() else a.checkpoint
        state=torch.load(selected,map_location='cpu',weights_only=True);model.load_state_dict(state['model_state'],strict=True)
    else:selected=a.checkpoint;state=ck
    threshold=.5;validation={}
    if binary:
        from risk_model import choose_threshold
        _,vy,vp=evaluate('val')
        try:validation=choose_threshold(vy,vp,a.precision_floor);threshold=validation['threshold']
        except ValueError as exc:raise ValueError('Validation selection failed; test set remains unused: '+str(exc)) from exc
    # Export a portable inference artifact, binding fitted preprocessing/tokenizer.
    inference={k:state[k] for k in ('model_state','config','epoch','seed')}
    inference['data_sha256']=sha(a.data);inference['validation_threshold']=validation
    prep=a.data.with_suffix('.preprocessing.json')
    if prep.exists():inference['preprocessing']={k:v for k,v in json.loads(prep.read_text()).items() if k!='split_indices'}
    vocab=a.data.with_suffix('.vocabulary.json')
    if vocab.exists():inference['vocabulary_sha256']=sha(vocab)
    torch.save(inference,a.output/'inference.pt')
    tl,y,p=evaluate('test');pred=(p>=threshold).astype(int) if binary else p.argmax(1)
    result={'protocol':'validation-selected checkpoint; one final held-out evaluation','config':config,'seed':a.seed,'selected_epoch':state['epoch'],'test':{**metrics(y,pred,classes),'loss':tl},'validation_threshold':validation,'data_sha256':sha(a.data),'checkpoint_sha256':sha(selected),'split_rows':{s:len(arrays['y_'+s]) for s in ('train','val','test')},'python':sys.version.split()[0],'numpy':np.__version__,'torch':torch.__version__,'evaluate_only':a.evaluate_only,'majority_class_baseline_accuracy':float((y==np.bincount(arrays['y_train'].astype(int),minlength=classes).argmax()).mean()),'inference_checkpoint_sha256':sha(a.output/'inference.pt')}
    result['training_arguments']=None if a.evaluate_only else {'total_epochs':a.epochs,'batch_size':a.batch_size,'learning_rate':a.lr,'optimizer':'AdamW','gradient_clip_norm':5,'resumed':a.resume,'seed':a.seed}
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    with (a.output/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['test_index','label','prediction','probabilities'])
        for i,(label,guess,prob) in enumerate(zip(y,pred,p)):w.writerow([i,int(label),int(guess),json.dumps(np.asarray(prob).tolist())])
    print(json.dumps(result),flush=True)
