import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import pairwise_distance, cosine_similarity
from torchvision.models import resnet50
from tqdm.notebook import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
workers = 0 if os.name == 'nt' else 8
print("DEVICE = ", device)
print("nworkers = ", workers)

data_dir = r".\lfw-py\DATA"
tiny_celeba = r".\tiny_celeba"

# CELEBA DATASET:
data_dir = r".\AUGMENTED"

# Create a dataset for triplet face images
class TripletFaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((96, 96)),
            transforms.Normalize(mean=[0], std=[1])
        ])
        self.classes = sorted(os.listdir(root_dir))

    def __getitem__(self, index):
        anchor_class = random.choice(self.classes)
        anchor_dir = os.path.join(self.root_dir, anchor_class)
        anchor_images = os.listdir(anchor_dir)
        anchor_image = random.choice(anchor_images)
        anchor_path = os.path.join(anchor_dir, anchor_image)
        anchor = self.transform(Image.open(anchor_path))

        positive_image = random.choice(anchor_images)
        while positive_image == anchor_image:
            positive_image = random.choice(anchor_images)
        positive_path = os.path.join(anchor_dir, positive_image)
        positive = self.transform(Image.open(positive_path))

        negative_class = random.choice(self.classes)
        while negative_class == anchor_class:
            negative_class = random.choice(self.classes)
        negative_dir = os.path.join(self.root_dir, negative_class)
        negative_images = os.listdir(negative_dir)
        negative_image = random.choice(negative_images)
        negative_path = os.path.join(negative_dir, negative_image)
        negative = self.transform(Image.open(negative_path))

        return anchor, positive, negative, anchor_dir

    def __len__(self):
        return len(self.classes)

# Create the DataLoader for training
tlfd = TripletFaceDataset(data_dir)

train_loader = DataLoader(tlfd,
                          num_workers=workers,
                          shuffle=True,
                          batch_size=32)

# Define the ResNet-based triplet network
class ResNet_Triplet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Feature_Extractor = resnet50(pretrained=True)
        num_filters = self.Feature_Extractor.fc.in_features
        self.Feature_Extractor.fc = nn.Sequential(
            nn.Linear(num_filters, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 10)
        )
        self.Triplet_Loss = nn.Sequential(
            nn.Linear(10, 2)
        )

    def forward(self, x):
        features = self.Feature_Extractor(x)
        triplets = self.Triplet_Loss(features)
        return triplets

# Define the triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Create an instance of the ResNet triplet network
ResNet = ResNet_Triplet()
ResNet = ResNet.to(device)
Optimizer = torch.optim.Adam(ResNet.parameters(), lr=0.0001)
criterion = TripletLoss()

# Train the model
def train(model, optimizer, EPOCHS, train_dl, output_model_name):
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(train_dl, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            Optimizer.zero_grad()
            anchor_out = ResNet(anchor_img)
            positive_out = ResNet(positive_img)
            negative_out = ResNet(negative_img)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            Optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
            print("Epoch: {}/{} — Loss: {:.4f}".format(epoch+1, EPOCHS, np.mean(running_loss)))

    torch.save(model, output_model_name)

train(ResNet, Optimizer, 10, train_loader, "RESNETCELEBA.pth")

# Function to display images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Function to plot loss
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
    plt.savefig("PLOT.jpg")




# COMPETITION/VALIDATION AND SUBMISSION SECTION

# Function to compare images
def compare_images(image_path, folder_path, distance_type, model_file):
    model = torch.load(model_file)
    model.eval()
    transform_input = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    transform_folder = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = transform_input(input_image).unsqueeze(0)
    results = []
    for filename in os.listdir(folder_path):
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
        result = [filename, distance.item()]
        results.append(result)
    results.sort(key=lambda x: x[1])
    res = results[:10]
    res1 = []
    for el in res:
        res1.append(el[0])
    return res1

# Define the data for submission
mydata = dict()
mydata['groupname'] = "Data biters with engineering spices"
ressubmit = dict()
gallery = r"C:\Users\npari\Desktop\test_data\gallery"    
queryfolder = r"C:\Users\npari\Desktop\test_data\query"
counter = 0
for query in os.listdir(r"C:\Users\npari\Desktop\test_data\query"):
    counter += 1
    print(counter, "/", len(os.listdir(queryfolder)))
    querypath = os.path.join(queryfolder, query)
    res = compare_images(querypath, queryfolder, "euclidean", "RESNETCELEBA.pth")
    ressubmit[query] = res

mydata["images"] = ressubmit

# Submit the results
import requests
import json

def submit(results, url="https://competition-production.up.railway.app/results/"): 
    res = json.dumps(results) 
    response = requests.post(url, res) 
    try: 
        result = json.loads(response.text) 
        print(f"accuracy is {result['results']}") 
        return result 
    except json.JSONDecodeError: 
        print(f"ERROR: {response.text}") 
        return None

submit(mydata)
