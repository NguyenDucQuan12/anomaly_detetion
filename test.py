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

# kiểm tra xem có sẵn GPU (CUDA) để sử dụng hay không. Nếu có, mô hình sẽ được đặt trên GPU
# để tăng tốc độ tính toán. Nếu không, nó sẽ sử dụng CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    #Chuyển đổi kích thước của ảnh thành (224, 224) pixel. Điều này thường
    # được thực hiện để chuẩn hóa kích thước của tất cả các ảnh trong tập dữ liệu.
    transforms.Resize((224,224)), # Unify images to (224,224) size
    # Chuyển đổi ảnh từ định dạng hình ảnh sang dạng tensor, để có thể sử dụng trong mô hình học máy
    transforms.ToTensor()
])

train_dir='./screw/train'
test_dir='./screw/test'

#Đây là đối tượng dữ liệu huấn luyện được tạo ra từ thư mục train_dir.
# Các ảnh trong thư mục này sẽ được xử lý bằng các biến đổi đã định nghĩa trong transform
train_dataset = ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#Đây là đối tượng DataLoader cho dữ liệu huấn luyện.
#Nó sẽ tải dữ liệu theo các lô (batch) có kích thước 16 và xáo trộn dữ liệu trước khi đưa vào mô hình
test_dataset = ImageFolder(root=test_dir, transform=transform)
#Tương tự, đây là đối tượng DataLoader cho dữ liệu kiểm tra.
# Không cần xáo trộn dữ liệu kiểm tra (shuffle=False) vì chúng ta muốn giữ nguyên thứ tự của các ảnh trong quá trình kiểm tra.
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# Đoạn code này lấy ra danh sách các lớp (classes) có trong tập dữ liệu kiểm tra.
# Đây có thể là các tên của các loại khuyết tật mà bạn muốn mô hình dự đoán.
class_names = test_dataset.classes
#print(class_names)

# Autoencoder
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

# Instantiate Autoencoder and send to GPU
autoencoder = Autoencoder().to(device)

# Definition of loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# training
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

# test
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

#### comparison bertween classes in test data
autoencoder.eval()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
class_encodings = [[] for i in range(6)]
with torch.no_grad():
    for data in test_loader:
        img, label = data
        img = Variable(img).to(device)
        encoding = autoencoder.encoder(img)
        class_encodings[label.item()].append(encoding.cpu().numpy().ravel())

class_means = []
class_mses = []
for i in range(6):
    class_means.append(np.mean(class_encodings[i]))
    class_mse = np.mean((class_encodings[i] - class_means[i])**2)
    class_mses.append(class_mse)

plt.figure(figsize=(8, 6))
plt.bar(np.arange(6)-0.2, class_means, width=0.4, label='Mean', alpha=0.5)
plt.bar(np.arange(6)+0.2, class_mses, width=0.4, label='MSE', alpha=0.5)
plt.xticks(range(6), class_names, rotation=90)
plt.legend()
plt.title("Class Encodings Mean and MSE Comparison")
plt.show()