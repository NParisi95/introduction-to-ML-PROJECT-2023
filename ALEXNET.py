import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.utils
from pprint import pprint
from torch.nn.functional import pairwise_distance, cosine_similarity

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
workers = 0 if os.name == 'nt' else 8
print("DEVICE = ", device)
print("nworkers = ", workers)

Directory containing the dataset
data_dir = r".\lfw-py\DATA"


# Custom dataset class for triplet data
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((227, 227)),
            transforms.Normalize(mean=[0], std=[1])])
        self.classes = sorted(os.listdir(root_dir))

    def __getitem__(self, index):
        # Select an anchor image from a random class
        anchor_class = random.choice(self.classes)
        anchor_dir = os.path.join(self.root_dir, anchor_class)
        anchor_images = os.listdir(anchor_dir)
        anchor_image = random.choice(anchor_images)
        anchor_path = os.path.join(anchor_dir, anchor_image)
        anchor = self.transform(Image.open(anchor_path))

        # Select a positive image from the same class as the anchor
        positive_image = random.choice(anchor_images)
        while positive_image == anchor_image:
            positive_image = random.choice(anchor_images)
        positive_path = os.path.join(anchor_dir, positive_image)
        positive = self.transform(Image.open(positive_path))

        # Select a negative image from a different class than the anchor
        negative_class = random.choice(self.classes)
        while negative_class == anchor_class:
            negative_class = random.choice(self.classes)
        negative_dir = os.path.join(self.root_dir, negative_class)
        negative_images = os.listdir(negative_dir)
        negative_image = random.choice(negative_images)
        negative_path = os.path.join(negative_dir, negative_image)
        negative = self.transform(Image.open(negative_path))

        return anchor, positive, negative

    def __len__(self):
        return len(self.classes)

# Create an instance of the TripletFaceDataset
tlfd = TripletFaceDataset(data_dir)

# Create a data loader for training
train_loader = DataLoader(tlfd,
                           num_workers=workers,
                           shuffle=True,
                           batch_size=8)

# Function to display images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Function to plot the loss
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
    plt.savefig("PLOT.jpg")


# Custom AlexNet model
class ImageRetrievalModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageRetrievalModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, p=2, dim=1)
        distance_negative = torch.norm(anchor - negative, p=2, dim=1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Create an instance of the model
model = ImageRetrievalModel().to(device)

# Define the triplet loss and optimizer
triplet_loss = TripletLoss(margin=0.2)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10

# Training loop
# def train(model, optimizer, num_epochs, train_loader):
#     loss_history = []
#     iteration = []
#     for epoch in range(num_epochs):
#         for batch_idx, (anchor_images, positive_images, negative_images) in enumerate(train_loader):
#             anchor_embeddings = model(anchor_images).to(device)
#             positive_embeddings = model(positive_images).to(device)
#             negative_embeddings = model(negative_images).to(device)

#             loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if batch_idx % 80 == 0:
#                 iteration.append(batch_idx)
#                 loss_history.append(loss.item)
#                 print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(anchor_images), len(train_loader.dataset),
#                     100. * batch_idx / len(train_loader), loss.item()))

#     torch.save(model, "TRIPLETCNNalexnet.pht")
#     show_plot(iteration, loss)

# Function to compare an input image with a gallery using cosine similarity or Euclidean distance
def compare_images(image_path, folder_path, distance_type, model_file):
    model = torch.load(model_file)
    model.eval()

    transform_input = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])

    transform_folder = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])

    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform_input(input_image).unsqueeze(0)

    results = []
    counter = 0
    for filename in os.listdir(folder_path):
        counter+=1
        if counter % 200 == 0:
            print(counter)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_file = os.path.join(folder_path, filename)
            image = Image.open(image_file).convert("RGB")
            tensor = transform_folder(image).unsqueeze(0)

            input_embedding = model(input_tensor)
            embedding = model(tensor)

            if distance_type == "euclidean":
                distance = pairwise_distance(input_embedding, embedding)
            elif distance_type == "cosine":
                distance = cosine_similarity(input_embedding, embedding)
            else:
                raise ValueError("Invalid distance type. Choose between 'euclidean' and 'cosine'.")

            result = [image_file, distance.item()]
            results.append(result)

    results.sort(key=lambda x: x[1])
    res = results[:10]

    for el in res:
        compare = Image.open(el[0]).convert("RGB")
        compare = transform_input(compare).unsqueeze(0)
        concatenated = torch.cat((input_tensor, compare), 0)
        imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {el[1]}')

    return res

# Specify the paths for the query image and gallery folder
gallery = r""    
query = r""

# Call the compare_images function with the specified parameters
res = (compare_images(query, gallery, "euclidean", "TRIPLETCNNalexnet.pht"))
pprint(res)






























