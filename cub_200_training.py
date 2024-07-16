import matplotlib.pyplot as plt
import os

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from cub_200_dataloader import load_data

def train(dataloader, model, loss_fn, optimizer, device, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    total_processed = 0
    train_loss = 0
    correct = 0

    model.train()

    for batch, (image, label, _) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)
        
        label = label.long() - 1

        pred = model(image).logits
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

        total_processed += len(image)

        # if total_processed % 2000 == 0:
        #     idx = torch.randint(0, len(X), (1,)).item()
        #     img = X[idx].cpu().permute(1, 2, 0)
        #     plt.imshow(img)
        #     plt.title(f"Epoch {epoch+1} - Batch {batch+1}")
        #     plt.axis('off')
        #     plt.show()
        
        print(f"Epoch {epoch+1:>3d} Batch {batch+1:>3d}: loss: {loss.item():>7f}  [{total_processed:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct /= size
    
    print(f"-------------------------------")
    print(f"Train Result:\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f}")

def test(dataloader, model, loss_fn, device, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    total_processed = 0
    test_loss = 0
    correct = 0

    model.eval()

    with torch.no_grad():
        for batch, (image, label, _) in enumerate(dataloader):
            image, label = image.to(device), label.to(device)
            
            label = label.long() - 1

            pred = model(image).logits
            loss = loss_fn(pred, label)

            test_loss += loss
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

            total_processed += len(image)

            print(f"Epoch {epoch+1:>3d} Test {batch+1:>3d}: loss: {test_loss / (batch + 1):>7f}  [{total_processed:>5d}/{size:>5d}]")

    test_loss /= num_batches
    correct /= size
    
    print(f"-------------------------------")
    print(f"Test Result:\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CUB_200_2011', 'cub_200_2011_dataset.csv')
    img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CUB_200_2011', 'CUB_200_2011')

    processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    # model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=200)
    # model = models.resnet18(weights='IMAGENET1K_V1')
    # model = ResNet18(num_classes=200)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.Grayscale(num_output_channels=3),
        ], p=0.3),
    ])

    image_transform = lambda image: processor(images=transform(image), return_tensors="pt")["pixel_values"].squeeze(0)

    train_loader, test_loader = load_data(csv_file, img_dir, batch_size=100, transform=image_transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 40

    for epoch in range(num_epochs):
        print(f"\n-------------------------------\nEpoch {epoch+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device, epoch)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs or (epoch + 1) == 1:
            print(f"\n-------------------------------")
            test(test_loader, model, loss_fn, device, epoch)

    print("Training complete")
