#! /usr/bin/python3

import argparse
import os

import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss
from torchvision import models
import torch.nn as nn

from dataset import AudioDataset
from utils import print_environment_info, provide_determinisim
from whistle_detection.utils import worker_seed_set

def run():
    print_environment_info()

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to dataset json file")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the batches")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory in which the checkpoints are stored")
    parser.add_argument("--train_test_split", type=float, default=0.8, help="Fraction of dataset to use for training")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Evaluation: Object confidence threshold")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    parser.add_argument("--sample_rate", type=int, default=10_000, help="Targeted sample rate of the audio files")
    parser.add_argument("--chunk_duration", type=int, default=1, help="Duration of the chunks in seconds")
    args = parser.parse_args()

    if args.seed != -1:
        provide_determinisim(args.seed)

    wandb.init(project="bitbots-whistle-detection")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = BCEWithLogitsLoss()

    train_dataset = AudioDataset(args.dataset_path, args.sample_rate, args.chunk_duration, train_mode=True, train_test_split=args.train_test_split, seed=args.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, worker_init_fn=worker_seed_set)

    validation_dataset = AudioDataset(args.dataset_path, args.sample_rate, args.chunk_duration, train_mode=False, train_test_split=args.train_test_split, seed=args.seed)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, worker_init_fn=worker_seed_set)

    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(
        params,
        lr=args.learning_rate,
    )

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs+1):
        model.train()

        for batch_i, (spectograms, labels) in enumerate(tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            spectograms = Variable(spectograms.to(device, non_blocking=True))
            labels = Variable(labels.float().to(device), requires_grad=False)

            outputs = model(spectograms).squeeze()

            loss = bce(outputs, labels)
            loss.backward()

            wandb.log({"train_loss": loss.item()})

            ###############
            # Run optimizer
            ###############

            optimizer.step()
            optimizer.zero_grad()

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            # Evaluate the model on the validation set
            metrics_output = {"mean": evaluate(model, validation_dataloader, args.conf_thres, device)}

            wandb.log(metrics_output)
            print(f'---- Evaluation metrics: {metrics_output} ----')
        


def evaluate(model, dataloader, conf_thres, device):
    model.eval()

    for spectograms, labels in tqdm.tqdm(dataloader, desc="Validating"):
        spectograms = Variable(spectograms.to(device), requires_grad=False)
        labels = Variable(labels.float().to(device), requires_grad=False)

        with torch.no_grad():
            outputs = model(spectograms).squeeze()

        return labels.eq(outputs >= conf_thres).float().mean()

if __name__ == "__main__":
    run()
