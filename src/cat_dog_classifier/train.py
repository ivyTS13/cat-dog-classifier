from pathlib import Path

import torch 
from torch import nn
from torch.optim import Adam

from cat_dog_classifier.dataset import create_dataloaders
from cat_dog_classifier.model import CatDogCNN


import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, train_loader,loss_function, optimizer, device):
    model.train()

    total_loss = 0.0
    correct =0
    total = 0

    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        #Make predictions
        outputs = model(images).squeeze(1)

        # Calculate how wrong the predictions are

        loss = loss_function(outputs,labels)

        # clear old gradients
        optimizer.zero_grad()

        # calculate gradients
        loss.backward()

        # update model weights
        optimizer.step()

        total_loss += loss.item()

        #convert logits to probabilities
        probabilities = torch.sigmoid(outputs)

        # Probability >= 0.5 -> dog (1)
        # Probability < 0.5  -> cat (0)
        predictions =(probabilities>=0.5).float()

        correct+= (predictions==labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(train_loader)

    accuracy = correct/total

    return average_loss,accuracy


def validate(model, val_loader, loss_function, device):

    model.eval()
    total_loss =0.0
    correct =0
    total =0

    with torch.no_grad():
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)

            loss = loss_function(outputs,labels)

            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities>=0.5).float()

            correct += (predictions==labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return average_loss, accuracy


def main():
    set_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("using device:", device)

    train_loader, val_loader,_ = create_dataloaders(
        "data",
        batch_size=32
    )

    model = CatDogCNN().to(device)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 20
    best_val_accuracy =0.0

    model_path = Path("models/cat_dog_cnn.pth")

    model_path.parent.mkdir(parents=True, exist_ok=True)


    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device  
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            loss_function,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy> best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                model_path
            )
            print("save new best model")


if __name__ == "__main__":
    main()
