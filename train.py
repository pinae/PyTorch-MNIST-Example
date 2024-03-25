import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from load_data import *
from MultiLayerPerceptron import FeedForwardNetwork, DeepFeedForwardNetwork
from ConvolutionalNetwork import ConvolutionalNetwork, DeepConvolutionalNetwork
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

def main():
    wandb.init(
        project="MNIST-Examples",

        config={
            "dataset": "MNIST",
            "learning_rate": 1e-0,
            "epochs": 40,
            "batch_size": 6000,
            "shuffle_data": True
        }
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    wandb.config["device"] = device

    train_dataloader = DataLoader(training_data(),
                                  batch_size=wandb.config["batch_size"],
                                  shuffle=wandb.config["shuffle_data"])
    test_dataloader = DataLoader(test_data(),
                                 batch_size=wandb.config["batch_size"],
                                 shuffle=wandb.config["shuffle_data"])
    for X, y in test_dataloader:
        input_size = X.shape[-1] * X.shape[-2]
        print(input_size)
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    #model = FeedForwardNetwork().to(device)
    model = DeepFeedForwardNetwork(input_size=input_size).to(device)
    #model = ConvolutionalNetwork().to(device)
    #model = DeepConvolutionalNetwork().to(device)
    print(model)

    loss_fn = CrossEntropyLoss()
    wandb.config["loss_fn"] = str(loss_fn)[:-2]

    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config["learning_rate"])
    wandb.config["optimizer"] = str(optimizer).split('(')[0].strip()


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            wandb.log({
                "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                "train_loss": loss.item(),
            })

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if True or batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        wandb.log({
            "avg_test_loss": test_loss,
            "accuracy": 100 * correct
        })
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    scheduler = StepLR(optimizer, step_size=2, gamma=0.75)
    for t in range(wandb.config["epochs"]):
        print(f"Epoch {t+1} (lr: {optimizer.state_dict()['param_groups'][0]['lr']})\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
        scheduler.step()
    wandb.finish()


if __name__ == "__main__":
    for i in range(10):
        main()
