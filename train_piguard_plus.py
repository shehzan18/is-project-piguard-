import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset

from PIGuard import PIGuard
from util import set_seed, get_logger, compute_accuracy
from params import parse_args
from eval import evaluate


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train():
    global logger
    args = parse_args()

    set_seed(args)
    logger = get_logger(os.path.join(args.logs, f"log_{args.name}.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info(f"  <<< {key}: {args.__dict__[key]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = args.epochs
    save_path = args.checkpoint_path

    os.makedirs(save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def preprocess_function(examples):
        encoding = tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

    data_files = {
        "train": args.train_set,
        "valid": args.valid_set,
    }

    dataset = load_dataset("json", data_files=data_files)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.map(
        lambda examples: {"labels": [label for label in examples["label"]]},
        batched=True
    )

    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    train_loader = DataLoader(
        encoded_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        encoded_dataset["valid"],
        batch_size=args.eval_batch_size,
        shuffle=False
    )

    model = PIGuard("microsoft/deberta-v3-base", num_labels=2, device=device)
    model.to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=epochs * len(train_loader)
    )

    criterion = FocalLoss(alpha=1.5, gamma=2.0)

    best_accuracy = -1
    best_model_path = None

    for epoch in range(epochs):
        model.train()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits, _ = model(
                input_ids,
                attention_mask,
                labels,
                mode="train"
            )

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()
            scheduler.step()

            if step % args.display == 0:
                logger.info(f"Step: {step} / {len(train_loader)}")
                logger.info(f"Loss: {loss:.4f}")

            should_eval = (
                ((step % args.save_step == 0) and (step != 0))
                or
                (step == len(train_loader) - 1)
            )

            if should_eval:
                model.eval()

                loss_list = []
                logits_list = []
                labels_list = []

                with torch.no_grad():
                    for eval_step, batch in enumerate(validation_loader):
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)

                        logits, _ = model(input_ids, attention_mask, labels)

                        loss = criterion(logits, labels)

                        loss_list.append(loss.cpu().item())
                        logits_list.append(logits.cpu())
                        labels_list.append(labels.cpu())

                        if eval_step % args.display == 0:
                            logger.info(f"Eval Step: {eval_step} / {len(validation_loader)}")
                            logger.info(f"Eval Loss: {loss:.4f}")

                combined_logits = torch.cat(logits_list, dim=0)
                combined_labels = torch.cat(labels_list, dim=0)

                pred = combined_logits.argmax(1)

                benign_accuracy, injection_accuracy, total_accuracy = compute_accuracy(
                    pred,
                    combined_labels
                )

                print(f"total accuracy: {total_accuracy}")
                print(f"benign accuracy: {benign_accuracy}")
                print(f"injection accuracy: {injection_accuracy}")
                print(f"loss: {np.mean(loss_list)}")

                if total_accuracy >= args.save_thres:
                    model_path = f"{save_path}/epoch_{epoch}_{step}_model.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved to {model_path}")

                    if total_accuracy > best_accuracy:
                        best_accuracy = total_accuracy
                        best_model_path = f"{save_path}/best_model.pth"
                        torch.save(model.state_dict(), best_model_path)
                        print(f"Saved to {best_model_path}")

                model.train()

    logger.info("Evaluate Best Model on Test Sets.")

    if best_model_path is not None:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"Loaded model from {best_model_path}")
        evaluate(model, args.dataset_root)
    else:
        logger.info("No best model checkpoint found.")


if __name__ == "__main__":
    train()