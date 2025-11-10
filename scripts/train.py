import os
import time
import json
import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch 
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.models import cifar10_net
from models.ensemble import Ensemble
from models.deit import deit_base_patch16_224

from torchvision.models import inception_v3, resnet101, densenet169, efficientnet_b4, mobilenet_v3_large
from torchvision.models import vgg16_bn

from datasets import get_dataset
from losses import ReducedCrossEntropyLoss, MixedCrossEntropyLoss, LinearLoss, PiecewiseLinearLoss

torch.backends.cudnn.benchmark = True

DeiT = deit_base_patch16_224(pretrained=True)
DeiT.head = nn.Linear(768, 3)
for n, p in DeiT.named_parameters():
    if "mlp" in n:
        p.requires_grad = True
    elif "head" in n:
        p.requires_grad = True
    else:
        p.requires_grad = False

# Function to create an argument parser
def create_parser():
    parser = ArgumentParser(description="Image classification with Bregman loss functions")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--n_schedule", type=list, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--log_running_loss", type=int, default=None)
    parser.add_argument("--cross_validation", type=bool, default=None)
    return parser

# Function to load the correct model 
def get_model(model_id, num_classes):
    MODELS = dict(VGG=vgg16_bn(num_classes=3),
                  ConvNetTest=cifar10_net(),
                  InceptionNet=inception_v3(num_classes=num_classes, pretrained=False), 
                  ResNet=resnet101(num_classes=3), 
                  DenseNet=densenet169(pretrained=True),
                  EfficientNet=efficientnet_b4(pretrained=True),
                  MobileNet=mobilenet_v3_large(num_classes=3),
                  Ensemble=Ensemble(n_classes=3, n_features=1024),
                  DeiT=DeiT)
    return MODELS.get(model_id)


# Get a timestamp
def millis():
    return str(round(time.time()))


def modify_model(model):
    for n, p in model.named_parameters():
        if "backbone" in n:
            p.requires_grad = False
    return model


# Create a directory
def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as error:
        print(error)


# Get the loss function
def get_criterion(identifier):
    criteria = dict(CrossEntropyLoss=nn.CrossEntropyLoss(label_smoothing=0.1), 
                    ReducedCrossEntropyLoss=ReducedCrossEntropyLoss(), 
                    MixedCrossEntropyLoss=MixedCrossEntropyLoss(), 
                    LinearLoss=LinearLoss(),
                    PiecewiseLinearLoss=PiecewiseLinearLoss())
    return criteria.get(identifier, nn.CrossEntropyLoss())


# Training loop
def train_loop(model, train_loader, criterion, optimizer, device, alpha=0):
    model.train()
    epoch_loss = []
    total_correct = 0
    total_samples = 0
    for (X, y) in tqdm.tqdm(train_loader, desc="Training"):
        X, y = X.to(device).float(), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        _, pred_labels = torch.max(pred, 1)
        if alpha != 0:
            loss = criterion(pred, y, alpha=alpha)
        else:
            loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += [loss.item()]
        total_correct += (pred_labels == y).sum().item()
        total_samples += y.size(0)
    return epoch_loss, total_correct, total_samples


# Validation loop
def test_loop(model, val_loader, criterion, device, desc, alpha=0):
    model.eval()
    val_loss = []
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for (X, y) in tqdm.tqdm(val_loader, desc=desc):
            X, y = X.to(device).float(), y.to(device)
            pred = model(X)
            _, pred_labels = torch.max(pred, 1)
            if alpha != 0:
                loss = criterion(pred, y, alpha=alpha)
            else:
                loss = criterion(pred, y)
            val_loss += [loss.item()]
            total_correct += (pred_labels == y).sum().item()
            total_samples += y.size(0)
    return val_loss, total_correct, total_samples


# Function to define a k-Fold dataset split for cross validation
def cross_validation_split(dataset, k):
    splits = []
    num_samples = len(dataset)
    num_val_samples = int(num_samples / k)

    # For each fold
    for i in range(k):
        trll = 0
        trlr = i * num_val_samples
        vall = trlr
        valr = i * num_val_samples + num_val_samples
        trrl = valr
        trrr = num_samples

        # Indices of the training set
        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))
        train_indices = train_left_indices + train_right_indices

        # Indices of the validation set
        val_indices = list(range(vall, valr))

        # Trainig and validation subsets
        trainset = Subset(dataset, train_indices)
        valset = Subset(dataset, val_indices)
        splits.append([trainset, valset])

    return splits


# Model training function
def train(model, trainset, valset, testset, hyperparameters, work_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Define a training log variable
    log = dict(hyperparameters=hyperparameters, 
               train_loss=[], 
               train_accuracy=[],
               val_loss=[],
               val_accuracy=[])

    # Unpack the training hyperparameters
    lr = hyperparameters["lr"]
    gamma = hyperparameters["gamma"]
    momentum = hyperparameters["momentum"]
    n_epochs = hyperparameters["n_epochs"]
    criterion = hyperparameters["criterion"]
    n_schedule = hyperparameters["n_schedule"]
    batch_size = hyperparameters["batch_size"]
    manual_seed = hyperparameters["manual_seed"]
    weight_decay = hyperparameters["weight_decay"]
    save_every = hyperparameters["save_every"]
    lr = lr * batch_size / 512

    # Split the dataset into training and validation and define the optimizer
    criterion = get_criterion(criterion)
    generator = torch.Generator().manual_seed(manual_seed)

    train_loader = DataLoader(trainset, batch_size=batch_size, generator=generator, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, generator=generator, num_workers=8, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    running_loss = []

    alphas = np.ones((n_epochs, ))

    max_val_accuracy = 0
    # Iterate over the whole dataset n_epoch times 
    for epoch in range(n_epochs):

        # Optimize, validate, and save 
        if hyperparameters["criterion"] == "CrossEntropyLoss":
            train_loss, train_total_correct, train_total_samples = train_loop(model, train_loader, criterion, optimizer, device)
            val_loss, val_total_correct, val_total_samples = test_loop(model, val_loader, criterion, device, "Validation")
        else:
            train_loss, train_total_correct, train_total_samples = train_loop(model, train_loader, criterion, optimizer, device, alpha=alphas[epoch])
            val_loss, val_total_correct, val_total_samples = test_loop(model, val_loader, criterion, device, "Validation", alpha=alphas[epoch])


        # Save the train log information
        running_loss += train_loss.copy()
        train_accuracy = train_total_correct / train_total_samples
        log["train_loss"].append(np.mean(train_loss))
        log["train_accuracy"].append(100 * train_accuracy)

        # Save the val log information
        val_accuracy = val_total_correct / val_total_samples
        log["val_loss"].append(np.mean(val_loss))
        log["val_accuracy"].append(100 * val_accuracy)

        # Print the training results per epoch
        log_string = "Epoch: {}\{} Train loss: {} Training accuracy: {} %  Val loss {} Val accuracy {} %"
        print(log_string.format(epoch + 1,
                                n_epochs, 
                                round(np.mean(train_loss), 3), 
                                round(100 * train_accuracy, 2),
                                round(np.mean(val_loss), 3),
                                round(100 * val_accuracy, 2)
                                ))

        # Append the txt log file per epoch
        with open(os.path.join(work_dir, "log.txt"), "a") as f:
            f.write(log_string.format(epoch + 1,
                                n_epochs, 
                                np.mean(train_loss), 
                                100 * train_accuracy,
                                np.mean(val_loss),
                                100 * val_accuracy
                                ) + "\n")

        # Save the model every few epochs
        if not (epoch + 1) % save_every:
            torch.save(model.state_dict(), os.path.join(work_dir, "epoch_" + str(epoch + 1) + ".pth"))
        
        # Update the learning rate
        scheduler.step()

        # Save the model with the highest validation accuracy
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(work_dir, "best.pth"))
    
    # Loss per iteration
    log["running_loss"] = running_loss

    # Test the model after training has finished
    model.load_state_dict(torch.load(os.path.join(work_dir, "best.pth")))
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    test_loss, test_total_correct, test_total_samples = test_loop(model, test_loader, criterion, device, "Testing", alpha=alphas[-1])
    log["test_loss"] = np.mean(test_loss)
    test_accuracy = test_total_correct / test_total_samples
    log["test_accuracy"] = 100 * test_accuracy

    # Print the testing results
    print("Test loss: {} Test accuracy: {} % " .format(round(np.mean(test_loss), 2), 
                                                       round(100 * test_accuracy, 2),
                                                        ))

    # Save the log file in json format
    with open(os.path.join(work_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=4)


# Function to load the training hyperparameters based on the config file
def load_config(args):

    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Load the training hyperparameters based on the config file 
    dataset_id = config["dataset_id"]
    hyperparameters = dict(lr=config["lr"], 
                           gamma=config["gamma"], 
                           momentum=config["momentum"], 
                           n_epochs=config["n_epochs"], 
                           batch_size=config["batch_size"], 
                           n_schedule=config["n_schedule"],
                           num_classes=config["num_classes"],
                           weight_decay=config["weight_decay"],
                           save_every=config["save_every"],
                           cross_validation=config["cross_validation"],
                           log_running_loss=config["log_running_loss"])

    # Overwrite the hyperparameters based on the command line arguments
    if args.lr:
        hyperparameters["lr"] = args.lr
    if args.gamma:
        hyperparameters['gamma'] = args.gamma
    if args.momentum:
        hyperparameters["momentum"] = args.momentum
    if args.n_epochs:
        hyperparameters["n_epochs"] = args.n_epochs
    if args.batch_size:
        hyperparameters["batch_size"] = args.batch_size
    if args.n_schedule:
        hyperparameters["n_schedule"] = args.n_schedule
    if args.weight_decay:
        hyperparameters["weight_decay"] = args.weight_decay
    if args.save_every:
        hyperparameters["save_every"] = args.save_every
    if args.cross_validation:
        hyperparameters["cross_validation"] = args.cross_validation
    if args.log_running_loss:
        hyperparameters["log_running_loss"] = args.log_running_loss

    return dataset_id, hyperparameters


# Main function
def main():

    # Parse the command line arguments and load the training hyperparameters
    parser = create_parser()
    args = parser.parse_args()
    dataset_id, hyperparameters = load_config(args)

    # Load the correct model and dataset
    model = get_model(args.model_id, hyperparameters["num_classes"])
    state_dict = model.state_dict()
    dataset, testset = get_dataset(dataset_id)

    # Define the dataset splits for cross validation
    if hyperparameters["cross_validation"]:
        num_random_trials = 10
        splits = cross_validation_split(dataset, num_random_trials)
    else:
        num_random_trials = 1
        splits = [dataset, testset]

    for i in range(num_random_trials):

        # Get the training and validation set
        if hyperparameters["cross_validation"]:
            trainset, valset = splits[i]
        else:
            trainset = dataset
            valset = testset

        # Define the random
        seed = np.random.randint(0, 100)
        timestamp = millis()

        # Train with CrossEntropyLoss
        work_dir = os.path.join("checkpoints", dataset_id, args.model_id, "CrossEntropyLoss", timestamp) 
        create_dir(work_dir)
        model.load_state_dict(state_dict)
        hyperparameters["manual_seed"] = seed
        hyperparameters["criterion"] = "CrossEntropyLoss"
        train(model, trainset, valset, testset, hyperparameters, work_dir)

        # Train with ReducedCrossEntropyLoss
        work_dir = os.path.join("checkpoints", dataset_id, args.model_id, "ReducedCrossEntropyLoss", timestamp) 
        print(work_dir)
        create_dir(work_dir)
        model.load_state_dict(state_dict)
        hyperparameters["manual_seed"] = seed
        hyperparameters["criterion"] = "ReducedCrossEntropyLoss"
        train(model, trainset, valset, testset, hyperparameters, work_dir)


if __name__ == "__main__":
    main()