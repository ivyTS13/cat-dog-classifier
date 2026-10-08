from pathlib import Path

import torch
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report

from cat_dog_classifier.dataset import create_dataloaders
from cat_dog_classifier.model import CatDogCNN


def main():
    # Decide whether to use CPU or CUDA
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Create test DataLoader
    _,_,test_loader = create_dataloaders(
        "data",
        batch_size=32
    )

    # Create the same model architecture
    model = CatDogCNN().to(device)

    # Load the best trained weights
    model_path = Path("models/cat_dog_cnn.pth")

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    # Put model into evaluation mode
    model.eval()

    loss_function = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    # We are ONLY evaluating, not training
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.float().to(device)

            # Make predictions
            outputs = model(images).squeeze(1)

            # Calculate loss
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            # Convert logits to probabilities
            probabilities = torch.sigmoid(outputs)

            # Convert probabilities to class predictions
            predictions = (probabilities >= 0.5).float()

            # Count correct predictions
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Save predictions and labels for confusion matrix
            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    # Calculate final metrics
    test_loss = total_loss / len(test_loader)
    test_accuracy = correct / total

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Confusion matrix
    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\nConfusion Matrix:")
    print(matrix)

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["cats", "dogs"]
        )
    )


if __name__ == "__main__":
    main()