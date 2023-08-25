import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize((224,224)), # Unify images to (224,224) size
    transforms.ToTensor()
])

train_dir='./screw/train'

test_dir='./screw/test'

train_dataset = ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = ImageFolder(root=test_dir, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class_names = test_dataset.classes
print(class_names)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001 )#là tốc độ học (learning rate)

for epoch in range(100):
    running_loss = 0.0
    for data in train_loader:
        img, _ = data
        img = Variable(img).to(device)
        optimizer.zero_grad()
        outputs = autoencoder(img)

        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('Epoch [%d], Loss: %.4f' % (epoch+1, running_loss/len(train_loader)))


autoencoder.eval()
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = Variable(img).to(device)
        outputs = autoencoder(img)
        loss = criterion(outputs, img)
        test_loss += loss.item()

print('Test Loss: %.4f' % (test_loss/len(test_loader)))


autoencoder.eval()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class_encodings = [[] for i in range(6)]
with torch.no_grad():
    for data in test_loader:
        img, label = data
        img = Variable(img).to(device)
        encoding = autoencoder.encoder(img)
        class_encodings[label.item()].append(encoding.cpu().numpy().ravel())


def detect_and_save_anomalies(autoencoder, test_loader, threshold, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    autoencoder.eval()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, _ = data
            img = Variable(img).to(device)
            outputs = autoencoder(img)
            loss = criterion(outputs, img)
            
            if loss.item() > threshold:
                img_filename = test_dataset.samples[i][0]
                img_basename = os.path.basename(img_filename)
                destination_path = os.path.join(save_dir, img_basename)
                shutil.copy(img_filename, destination_path)
                
anomaly_threshold = (test_loss/len(test_loader))
anomaly_save_dir = 'anomaly_images'

detect_and_save_anomalies(autoencoder, test_loader, anomaly_threshold, anomaly_save_dir)
print("Phát hiện và lưu ảnh có bất thường hoàn thành.")

