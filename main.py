import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from custom_dataset import CactiDataset
from imshow_func import imshow
from cnn_model import CNN


# place the files in your IDE working dicrectory .
labels = pd.read_csv('aerial-cactus-identification/train.csv')
submission = pd.read_csv('aerial-cactus-identification/sample_submission.csv')
print(labels.head())
print(labels['has_cactus'].value_counts())
# Exploring label distribution
#As per the pie chart, the data is biased towards one class. Imbalanced data will affect the final results. We already have enough data for CNN to produce results, so there is no need for any data sampling or augmentation.
label = 'Has Cactus', 'Hasn\'t Cactus'
plt.figure(figsize = (8,8))
plt.pie(labels.groupby('has_cactus').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
#plt.show()


train_path = 'aerial-cactus-identification/train'
test_path = 'aerial-cactus-identification/test'

# Image Pre-processing
#Images in a dataset do not usually have the same pixel intensity and dimensions. In this section, you will pre-process the dataset by standardizing the pixel values.
#The next required process is transforming raw images into tensors so that the algorithm can process them.

fig, ax = plt.subplots(1, 5, figsize=(15, 3))

for i, idx in enumerate(labels[labels['has_cactus'] == 1]['id'][-5:]):
    path = os.path.join(train_path, idx)
    ax[i].imshow(img.imread(path))

# Normalization
means = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, std)])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, std)])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, std)])

# You can stack multiple image transformation commands in transform.Compose.
# Normalizing an image is an important step that makes model training stable and fast. In tranforms.Normalize() class, a list of means and standard deviations is sent in the form of a list.
train, valid_data = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)
train_data = CactiDataset(train, train_path, train_transform)
valid_data = CactiDataset(valid_data, train_path, valid_transform)
test_data = CactiDataset(submission, test_path, test_transform)


# Hyper parameters

#num_epochs = 35
num_epochs = 1
num_classes = 2
batch_size = 25
learning_rate = 0.001

# CPU or GPU

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset = valid_data, batch_size = batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle=False, num_workers=0)

# Visualize normalized train images
trainimages,trainlabels = next(iter(train_loader))

fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
print('Display 5 training images')
for i in range(5):
    axe1 = axes[i]
    imshow(trainimages[i], ax=axe1, normalize=False)
plt.title('After normalized train images:')
#plt.show()
print(trainimages[0].size())

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
print('CNN architecture:/n',model)

# keeping-track-of-losses
train_losses = []
valid_losses = []

for epoch in range(1, num_epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0

    # training-the-model
    model.train()
    for data, target in train_loader:
        # move-tensors-to-GPU
        data = data.to(device)
        target = target.to(device)

        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-single-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)

    # validate-the-model
    model.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

# Visualize train and validation losses
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()



# test-the-model
model.eval()  # it-disables-dropout
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

model.eval()  # it-disables-dropout
finals = []
with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        finals.append(predicted)
print(finals[0])

# Save
torch.save(model.state_dict(), 'model.ckpt')