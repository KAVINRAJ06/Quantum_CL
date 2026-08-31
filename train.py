import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from quantum_sam import QuantumSAMSegmenter, EWC, ReplayBuffer
from quantum_sam.config import load_config, dataset_options
from quantum_sam.data import PairedAerialDataset, discover_pairs, infer_palette

def miou(logits, masks, classes):
    predictions, values = logits.argmax(1), []
    for cls in range(classes):
        intersection = ((predictions == cls) & (masks == cls)).sum().float()
        union = ((predictions == cls) | (masks == cls)).sum().float()
        if union > 0: values.append(intersection / union)
    return torch.stack(values).mean().item() if values else 0.0

def make_loader(data, split, args, shuffle=False):
    allowed = {"images", "masks", "image_suffix", "mask_suffix", "recursive", "palette"}
    kwargs = {key: value for key, value in data.items() if key in allowed}
    dataset = PairedAerialDataset(data["root"], split, args.image_size, **kwargs)
    if dataset.palette and len(dataset.palette) > args.num_classes:
        raise ValueError(f"Detected {len(dataset.palette)} mask classes but model has num_classes={args.num_classes}.")
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=torch.cuda.is_available())

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = QuantumSAMSegmenter(args.num_classes, args.sam_model, args.qubits, args.freeze_sam).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    criterion, ewc, replay = nn.CrossEntropyLoss(ignore_index=255), EWC(model, args.ewc_strength), ReplayBuffer(args.replay_size)
    task_data = [(args.tasks[0], dataset_options(load_config(args.config)))] if args.config else [(task, {"root": str(Path(args.data_root) / task)}) for task in args.tasks]
    for task, data in task_data:
        # Infer RGB colours once across both splits so class indices remain identical.
        if data.get("palette", "auto") == "auto":
            pair_args = (data.get("images", "images/{split}"), data.get("masks", "masks/{split}"), data.get("image_suffix", ""), data.get("mask_suffix", ""), data.get("recursive", True))
            all_pairs = discover_pairs(data["root"], "train", *pair_args)
            all_pairs += discover_pairs(data["root"], "val", *pair_args)
            data["palette"] = infer_palette(all_pairs)
        train, val = make_loader(data, "train", args, True), make_loader(data, "val", args)
        print(f"{task}: {len(train.dataset)} train / {len(val.dataset)} val pairs")
        for epoch in range(args.epochs_per_task):
            model.train()
            for images, masks in tqdm(train, desc=f"{task} {epoch + 1}/{args.epochs_per_task}"):
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                loss = criterion(model(images), masks) + ewc.penalty()
                if len(replay):
                    ri, rm = replay.sample(args.replay_batch, device)
                    loss = loss + args.replay_weight * criterion(model(ri), rm)
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); replay.add(images, masks)
            model.eval(); scores = []
            with torch.no_grad():
                for images, masks in val:
                    scores.append(miou(model(images.to(device, non_blocking=True)), masks.to(device, non_blocking=True), args.num_classes))
            print(f"{task}: epoch {epoch + 1} val mIoU={sum(scores) / max(len(scores), 1):.4f}")
        ewc.consolidate(train, criterion, device)
        torch.save({"model": model.state_dict(), "task": task, "args": vars(args)}, f"checkpoint_{task}.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Configurable Quantum-SAM segmentation training")
    p.add_argument("--config", help="YAML dataset/training configuration; preferred")
    p.add_argument("--data-root", default="data"); p.add_argument("--tasks", nargs="+", default=["openearthmap"])
    p.add_argument("--num-classes", type=int, default=8); p.add_argument("--sam-model", default="facebook/sam-vit-base")
    p.add_argument("--qubits", type=int, default=8); p.add_argument("--freeze-sam", action="store_true")
    p.add_argument("--image-size", type=int, default=512); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs-per-task", type=int, default=15); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ewc-strength", type=float, default=10); p.add_argument("--replay-size", type=int, default=256)
    p.add_argument("--replay-batch", type=int, default=1); p.add_argument("--replay-weight", type=float, default=.5); p.add_argument("--workers", type=int, default=0)
    parsed = p.parse_args()
    if parsed.config:
        cfg = load_config(parsed.config)
        train_cfg = cfg.get("training", {})
        for key, value in train_cfg.items():
            if hasattr(parsed, key): setattr(parsed, key, value)
        if "num_classes" in cfg: parsed.num_classes = cfg["num_classes"]
        if "sam_model" in cfg: parsed.sam_model = cfg["sam_model"]
        if "freeze_sam" in cfg: parsed.freeze_sam = cfg["freeze_sam"]
        parsed.tasks = [Path(parsed.config).stem]
    main(parsed)