import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np

from CNN_model  import DigitCNN
from data_loader import load_train_dataframe, get_train_image_path
from preprocessing import load_grayscale_image, preprocess_image
from config import IMAGE_SIZE
from CNN_evaluate import plot_confusion_matrix
import torch

#load the data
df = load_train_dataframe()

X = [] #images
y = [] #labels



data = torch.load("models/predictions.pth", weights_only=False)

y_true = data["y_true"]
y_pred = data["y_pred"]

plot_confusion_matrix(y_true, y_pred)
#loop over all samples
for _, row in df.iterrows():
    image_id = row["Id"]
    label = row["Category"]
    #get image path
    image_path = get_train_image_path(image_id, label)
    #load and preprocess image
    image = load_grayscale_image(image_path)
    image = preprocess_image(
        image,
        target_size=IMAGE_SIZE,
        use_crop=True,
        use_normalize=True
    )
    #store data
    X.append(image)
    y.append(label)

#convert to numpy array
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
#convert to pytorch tensors
X = torch.tensor(X)
y = torch.tensor(y)

#add channel dimension for CNN
X = X.unsqueeze(1)  #shape:(N, 1, H, W)

#combine data into dataset
dataset = TensorDataset(X, y)
#split into training and validation set (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])
#create batches
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

model = DigitCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

os.makedirs("models", exist_ok=True)

for epoch in range(10):
    model.train() #set model to training mode
    total_loss = 0

    for X_batch, y_batch in train_loader:
        #forward pass (prediction)
        outputs = model(X_batch)
        #compute loss
        loss = criterion(outputs, y_batch)
        #backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval() #evaluate model
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            #get predicted class
            _, preds = torch.max(outputs, 1)

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            #store predictions
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())

    val_acc = correct / total
    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}, Val Acc: {val_acc:.4f}")

#save trained model
torch.save(model.state_dict(), "models/cnn_model.pth")

#save predictions for evaluation
torch.save({
    "y_true": y_true,
    "y_pred": y_pred
}, "models/predictions.pth")

print("Model and predictions saved!")