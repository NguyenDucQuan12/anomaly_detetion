import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import os
import shutil
import datetime

# Trong lần chạy đầu tiên phải để load_model= false, bởi sau lần chạy
# đầu tiên thì mới tạo được file checkpoint cho các lần tiếp theo sử dụng
number_epoch = 10
number_epoch_train = 80
load_model = True
# Sau quá trình thay đổi ngưỡng bất thường không đạt kết quả tốt,
# thay đổi lr=0,01 cho kết quả khả quan, tỉ lệ đúng nhiều hơn
# lr = 0.001
lr = 0.001

now = datetime.datetime.now()
today = str(now.strftime("%d-%m-%y %Hh%Mp%Ss"))
date_today = str(now.strftime("%d-%m-%y"))
filename = "my_checkpoint_"+date_today
filename_direct = './file_checkpoint/'+filename


def save_checkpoint(state, filename=filename_direct):
    print("Save checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Unify images to (224,224) size
    transforms.ToTensor()
])

train_dir = './screw/train'

test_dir = './screw/test'

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
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # là tốc độ học (learning rate)

if load_model:
    load_checkpoint(torch.load("file_checkpoint/"+filename))

for epoch in range(number_epoch):
    running_loss = 0.0
    if epoch % 3 == 0:
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        outputs = model(img)

        loss = criterion(outputs, img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('Epoch [%d], Loss: %.4f' % (epoch + 1, running_loss / len(train_loader)))

model.eval()
test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = img.to(device)
        outputs = model(img)
        loss = criterion(outputs, img)
        test_loss += loss.item()

print('Test Loss: %.4f' % (test_loss / len(test_loader)))

model.eval()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class_encodings = [[] for i in range(6)]
with torch.no_grad():
    for data in test_loader:
        img, label = data
        img = img.to(device)
        encoding = model.encoder(img)
        class_encodings[label.item()].append(encoding.cpu().numpy().ravel())


def detect_and_save_anomalies(model, test_loader, threshold, save_dir_anomaly, save_dir_normal):
    os.makedirs(save_dir_normal, exist_ok=True)
    os.makedirs(save_dir_anomaly, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, _ = data
            img = img.to(device)
            outputs = model(img)
            loss = criterion(outputs, img)

            if loss.item() > threshold:
                img_filename = test_dataset.samples[i][0]
                img_basename = os.path.basename(img_filename)
                destination_path = os.path.join(save_dir_anomaly, img_basename)
                shutil.copy(img_filename, destination_path)
                print("anomaly"+str(i) + ": "+img_filename)

            else:
                img_filename = test_dataset.samples[i][0]
                img_basename = os.path.basename(img_filename)
                destination_path = os.path.join(save_dir_normal, img_basename)
                shutil.copy(img_filename, destination_path)
                print("normal"+str(i) + ": "+img_filename)


# Thay đổi class Autoencoder bằng cách thêm 1 lớp mạng nữa, lr= 0,001, kết quả normal
# anomaly_threshold = 0.05
# Thay đổi lần 6, kết quả hoàn hảo, sai 10 ảnh trong 160 ảnh
anomaly_threshold = 0.045
name_folder_anomaly = "anomaly_images"
name_folder_anomaly = today + " " + name_folder_anomaly

name_folder_normal = "normal_images"
name_folder_normal = today + " " + name_folder_normal
detect_and_save_anomalies(model, test_loader, anomaly_threshold, name_folder_anomaly, name_folder_normal)
print("Phát hiện và lưu ảnh có bất thường hoàn thành.")
