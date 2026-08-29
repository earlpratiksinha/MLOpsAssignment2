import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

# Force CPU if MPS creates memory pressure on Mac OS
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load 100% of processed datasets
    train_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(PROCESSED_DIR, "val"), transform=transform)

    # Set persistent_workers=False and num_workers=0 to eliminate memory leaks
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    mlflow.set_experiment("Cat_Dog_Classification_Full")

    with mlflow.start_run():
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("lr", LR)
        mlflow.log_param("model_architecture", "MobileNetV2")
        mlflow.log_param("dataset_size", len(train_dataset))

        for epoch in range(EPOCHS):
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            v_loss = val_loss / val_total
            v_acc = val_correct / val_total

            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}")

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", v_loss, step=epoch)
            mlflow.log_metric("val_acc", v_acc, step=epoch)

            # Clear memory caches between epochs
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        model_path = os.path.join(MODEL_DIR, "best_model.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")
        print(f"Full dataset training complete! Model saved to {model_path}")

if __name__ == "__main__":
    main()
