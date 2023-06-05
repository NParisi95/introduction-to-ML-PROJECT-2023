from torchvision.datasets import LFWPeople
import numpy as np
import os
import random
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image

# Set the number of workers for data loading
workers = 0 if os.name == 'nt' else 8
# Set the device to run the code on (GPU if available, otherwise CPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# DOWNLOAD DATASET
def download_dataset():
    # Once downloaded, extract the file called "lfw-funneled"
    # and rename the extracted folder as "DATA"
    dataset = LFWPeople("./", split="10fold", download=True)
    return dataset

# Check the file path where you are
work_dir = os.path.abspath("")
dataset_dir = os.path.join(work_dir, "CELEBA/PARSED_DATA")

class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        
    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # We need approximately 50% of the images to be in the same class
        should_get_same_class = random.randint(0, 1) 
        if should_get_same_class:
            while True:
                # Look until the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # Look until a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function, we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    


# ALTERNATIVE MODEL          
# class VGG16(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGG16, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(), 
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU())
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer6 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU())
#         self.layer7 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer8 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer9 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer10 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.layer11 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer12 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU())
#         self.layer13 = nn.Sequential(
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2, stride = 2))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(7*7*512, 4096),
#             nn.ReLU())
#         self.fc1 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU())
#         self.fc2= nn.Sequential(
#             nn.Linear(4096, 2))
        
#     def forward_once(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = self.layer8(out)
#         out = self.layer9(out)
#         out = self.layer10(out)
#         out = self.layer11(out)
#         out = self.layer12(out)
#         out = self.layer13(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         out = self.fc1(out)
#         out = self.fc2(out)
#         return out   
    
#     def forward(self, x, y):
#         output1 = self.forward_once(x)
#         output2 = self.forward_once(y)
#         return output1, output2

            
            




class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate the Euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive  

net = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

def train(train_dataloader, net, criterion, optimizer, epochs):
    counter = []
    loss_history = []
    iteration_number = 0
    
    for epoch in range(epochs):
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):
            img0, img1, label = img0.to(device), img1.to(device), label.to(device) 
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            
            if i % 10 == 0 :
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    
    torch.save(net, "contrastive_model.pht")

TRAIN_PATH = os.path.join(work_dir, "/TRAIN")
VALIDATION_PATH = os.path.join(work_dir, "/VALIDATION")

x = ImageFolder(TRAIN_PATH)
y = ImageFolder(VALIDATION_PATH)

# Set up transformations for the dataset
transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1]),
    transforms.Resize((100, 100))
])    

# Setup for VGG16
transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])  

siamesetraindataset = SiameseNetworkDataset(imageFolderDataset=x, transform=transformation)
train_dataloader = DataLoader(siamesetraindataset,
                              shuffle=True,
                              num_workers=workers,
                              batch_size=8)

train(train_dataloader, net, criterion, optimizer, 10)

siamesevalidationdataset = SiameseNetworkDataset(imageFolderDataset=y, transform=transformation)
validation_dataloader = DataLoader(siamesevalidationdataset,
                                   batch_size=1)

import matplotlib.pyplot as plt

train(train_dataloader, net, criterion, optimizer, 10)

# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

# Plotting data
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
    plt.savefig("PLOT.jpg")
    
# Take 1 photo to compare with other n photos in order to measure how the model works
def take_a_look(validation_dataloader):
    net = torch.load("model.pht")
    dataiter = iter(validation_dataloader)
    x0, _, _ = next(dataiter)
    
    for i in range(10):
        # Iterate over 5 images and test them with the first image (x0)
        _, x1, label2 = next(dataiter)
    
        # Concatenate the two images together
        concatenated = torch.cat((x0, x1), 0)
        
        output1, output2 = net(x0.to(device), x1.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)
        imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')

# Evaluate the model
def image_retrieval(gallery):
    res = []
    net = torch.load("model.pht")
    dataiter = iter(gallery)
    x0, _, _ = next(dataiter)
    counter = 0
    while dataiter:
        counter += 1
        try:
            _, x1, label2 = next(dataiter)
        except:
            break
    
        output1, output2 = net(x0.cpu(), x1.cpu())
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        res.append((float(euclidean_distance.item()), x1))
        
        if counter % 500 == 0:
            print(f"{counter} element parsed \n ======")
            
    res.sort(key=lambda x: x[0])
    
    res = res[:10]

    for el in res:
        concatenated = torch.cat((x0, el[1]), 0)
        imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {el[0]}')

train_dataloader = DataLoader(siamesetraindataset, batch_size=1)

image_retrieval(validation_dataloader)
