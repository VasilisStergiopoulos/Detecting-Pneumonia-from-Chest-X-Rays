import os
import glob
import json
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Function to display the training accuracy per epoch
def display_train_val_accuracy(dataset_id):

    # Cross Entropy Accuracy
    train_acc_ce = list()
    val_acc_ce = list()
    directories_ce = glob.glob(os.path.join("checkpoints", dataset_id, "CrossEntropyLoss", "*"))
    for dir in directories_ce:
        with open(os.path.join(dir, "log.json"), "r") as f:
            data = json.load(f)
            train_acc_ce.append(np.expand_dims(100 - np.array(data["train_accuracy"]), 0))
            val_acc_ce.append(np.expand_dims(100 - np.array(data["val_accuracy"]), 0))

    # Reduced Cross Entropy Accuracy
    train_acc_rce = list()
    val_acc_rce = list()
    directories_rce = glob.glob(os.path.join("checkpoints", dataset_id, "ReducedCrossEntropyLoss", "*"))
    for dir in directories_rce:
        with open(os.path.join(dir, "log.json"), "r") as f:
            data = json.load(f)
            train_acc_rce.append(np.expand_dims(100 - np.array(data["train_accuracy"]), 0))
            val_acc_rce.append(np.expand_dims(100 - np.array(data["val_accuracy"]), 0))

    train_acc_ce = np.concatenate(train_acc_ce)
    train_acc_rce = np.concatenate(train_acc_rce)
    val_acc_ce = np.concatenate(val_acc_ce)
    val_acc_rce = np.concatenate(val_acc_rce)

    # Get the mean and std data
    mean_train_acc_ce = np.mean(train_acc_ce, axis=0)
    std_train_acc_ce = np.std(train_acc_ce, axis=0)
    mean_train_acc_rce = np.mean(train_acc_rce, axis=0)
    std_train_acc_rce = np.std(train_acc_rce, axis=0)
    mean_val_acc_ce = np.mean(val_acc_ce, axis=0)
    std_val_acc_ce = np.std(val_acc_ce, axis=0)
    mean_val_acc_rce = np.mean(val_acc_rce, axis=0)
    std_val_acc_rce = np.std(val_acc_rce, axis=0)
    
    # Display the results
    epochs = np.arange(0, len(mean_train_acc_ce))
    plt.style.use('ggplot')
    
    # Training accuracy
    plt.figure()
    plt.plot(epochs, mean_train_acc_ce, label="CrossEntropy")
    plt.fill_between(epochs, mean_train_acc_ce - std_train_acc_ce, mean_train_acc_ce + std_train_acc_ce, alpha=0.4)
    plt.plot(epochs, mean_train_acc_rce, label="ReducedCrossEntropy")
    plt.fill_between(epochs, mean_train_acc_rce - std_train_acc_rce, mean_train_acc_rce + std_train_acc_rce, alpha=0.4)
    plt.xticks(np.arange(0, 52, 4))
    plt.title("Training Error")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    # Validation accuracy
    plt.figure()
    plt.plot(epochs, mean_val_acc_ce, label="CrossEntropy")
    plt.fill_between(epochs, mean_val_acc_ce - std_val_acc_ce, mean_val_acc_ce + std_val_acc_ce, alpha=0.4)
    plt.plot(epochs, mean_val_acc_rce, label="ReducedCrossEntropy")
    plt.fill_between(epochs, mean_val_acc_rce - std_val_acc_rce, mean_val_acc_rce + std_val_acc_rce, alpha=0.4)
    plt.xticks(np.arange(0, 52, 4))
    plt.title("Validation Error")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


# Function to display the testing accuracy as a barplot
def display_test_accuracy_box(dataset_id):
    
    test_acc_ce = list()
    test_acc_rce = list()

    # Cross Entropy Test Accuracy
    directories_ce = glob.glob(os.path.join("checkpoints", dataset_id, "CrossEntropyLoss", "*"))
    for dir in directories_ce:
        with open(os.path.join(dir, "log.json"), "r") as f:
            data = json.load(f)
            test_acc_ce.append(100 - data["test_accuracy"])

    # Reduced Cross Entropy Test Accuracy
    directories_rce = glob.glob(os.path.join("checkpoints", dataset_id, "ReducedCrossEntropyLoss", "*"))
    for dir in directories_rce:
        with open(os.path.join(dir, "log.json"), "r") as f:
            data = json.load(f)
            test_acc_rce.append(100 - data["test_accuracy"])

    print(np.mean(test_acc_ce))
    print(np.mean(test_acc_rce))
    
    plot_data = {"CrossEntropy": test_acc_ce, "ReducedCrossEntropy": test_acc_rce}

    plt.figure()
    sns.set_theme(style="darkgrid")
    ax = sns.boxplot(data=plot_data)
    plt.title("Testing Error")

    # adding transparency to colors
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .3))
    plt.show()