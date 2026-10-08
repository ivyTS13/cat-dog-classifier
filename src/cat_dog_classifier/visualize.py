import matplotlib.pyplot as plt

from cat_dog_classifier.dataset import create_dataloaders


def main():
    train_loader,_ = create_dataloaders(
        "data",
        batch_size=8
    )

    images, labels = next(iter(train_loader))

    class_name = train_loader.dataset.classes
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for image, label,ax in zip(images, labels, axes.flat):
        image = image.permute(1,2,0)

        ax.imshow(image)
        ax.set_title(class_name[label.item()])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()