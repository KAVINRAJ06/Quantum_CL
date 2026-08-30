import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from quantum_sam import QuantumSAMSegmenter, EWC, ReplayBuffer
from quantum_sam.data import PairedAerialDataset

def miou(logits, masks, classes):
    predictions = logits.argmax(1); values=[]
    for cls in range(classes):
        intersection=((predictions==cls)&(masks==cls)).sum().float(); union=((predictions==cls)|(masks==cls)).sum().float()
        if union: values.append(intersection/union)
    return torch.stack(values).mean().item() if values else 0.0

def main(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=QuantumSAMSegmenter(args.num_classes,args.sam_model,args.qubits,args.freeze_sam).to(device)
    optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=args.lr,weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss(ignore_index=255); ewc=EWC(model,args.ewc_strength); replay=ReplayBuffer(args.replay_size)
    for task in args.tasks:
        train=DataLoader(PairedAerialDataset(Path(args.data_root)/task,'train',args.image_size),batch_size=args.batch_size,shuffle=True,num_workers=args.workers)
        val=DataLoader(PairedAerialDataset(Path(args.data_root)/task,'val',args.image_size),batch_size=args.batch_size,num_workers=args.workers)
        for epoch in range(args.epochs_per_task):
            model.train()
            for images,masks in tqdm(train,desc=f'{task} {epoch+1}/{args.epochs_per_task}'):
                images,masks=images.to(device),masks.to(device); logits=model(images); loss=criterion(logits,masks)+ewc.penalty()
                if len(replay):
                    ri,rm=replay.sample(args.replay_batch,device); loss=loss+args.replay_weight*criterion(model(ri),rm)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); replay.add(images,masks)
            model.eval(); scores=[]
            with torch.no_grad():
                for images,masks in val: scores.append(miou(model(images.to(device)),masks.to(device),args.num_classes))
            print(f'{task}: val mIoU={sum(scores)/len(scores):.4f}')
        ewc.consolidate(train,criterion,device)
        torch.save({'model':model.state_dict(),'task':task,'args':vars(args)},f'checkpoint_{task}.pt')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='data'); p.add_argument('--tasks',nargs='+',default=['openearthmap','loveda']); p.add_argument('--num-classes',type=int,default=8); p.add_argument('--sam-model',default='facebook/sam-vit-base'); p.add_argument('--qubits',type=int,default=8); p.add_argument('--freeze-sam',action='store_true'); p.add_argument('--image-size',type=int,default=512); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--epochs-per-task',type=int,default=15); p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--ewc-strength',type=float,default=10); p.add_argument('--replay-size',type=int,default=256); p.add_argument('--replay-batch',type=int,default=1); p.add_argument('--replay-weight',type=float,default=.5); p.add_argument('--workers',type=int,default=0); main(p.parse_args())
