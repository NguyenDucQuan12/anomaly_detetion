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
    #Chuyển đổi kích thước của ảnh thành (224, 224) pixel
    #thực hiện để chuẩn hóa kích thước của tất cả các ảnh trong tập dữ liệu.
    transforms.Resize((224,224)), # Unify images to (224,224) size
    # Chuyển đổi ảnh từ định dạng hình ảnh sang dạng tensor, để có thể sử dụng trong mô hình học máy
    transforms.ToTensor()
])

train_dir='./screw/train'

test_dir='./screw/test'

#dữ liệu huấn luyện được tạo ra từ thư mục train_dir.
#ảnh trong thư mục sẽ được xử lý bằng các biến đổi đã định nghĩa trong transform
train_dataset = ImageFolder(root=train_dir, transform=transform)
#là đối tượng DataLoader cho dữ liệu huấn luyện.
#tải dữ liệu theo các lô (batch) có kích thước 16 và xáo trộn dữ liệu trước khi đưa vào mô hình
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#Tương tự, đây là đối tượng DataLoader cho dữ liệu kiểm tra.
#Không cần xáo trộn dữ liệu kiểm tra (shuffle=False) vì muốn giữ nguyên thứ tự của các ảnh trong quá trình kiểm tra.
test_dataset = ImageFolder(root=test_dir, transform=transform)

#ấy ra danh sách các lớp (classes) có trong tập dữ liệu kiểm tra.
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#là các tên của các loại khuyết tật mà bạn muốn mô hình dự đoán.
#Phải được định nghĩa trong thư mục test
class_names = test_dataset.classes
#in ra thử xem có dự đoán chính xác tên không
#print(class_names)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        #tạo một chuỗi liên tiếp các lớp mạng, nghĩa là dữ liệu sẽ chạy qua từng lớp một theo thứ tự
        #Bộ mã hóa này có nhiều lớp Conv2d và các phép biến đổi để
        # giảm kích thước của dữ liệu ảnh và tạo ra biểu diễn nén có số chiều thấp hơn, chứa các đặc trưng quan trọng của ảnh gốc.
        self.encoder = nn.Sequential(
            #Lớp này là một lớp Conv2d (convolutional layer) với 3 kênh đầu vào (3 là số kênh của ảnh màu RGB), 16 kênh đầu ra,
            # kernel_size=3 là kích thước của kernel convolution là 3x3, và padding=1 là sử dụng đệm để bảo toàn kích thước ảnh sau khi qua lớp này.
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            #Lớp ReLU activation thực hiện hàm kích hoạt ReLU trên đầu ra của lớp trước đó.
            nn.ReLU(),
            #Lớp MaxPool2d thực hiện lấy giá trị lớn nhất
            # trong mỗi vùng 2x2 của đầu ra trước đó và giảm kích thước ảnh đi một nửa (kích thước vùng và bước dịch chuyển đều là 2).
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential( #nn.Sequential(...) để tạo một chuỗi liên tiếp các lớp mạng
            #Lớp ConvTranspose2d thực hiện việc ngược lại so với Conv2d, giúp tăng kích thước của dữ liệu
            #output_padding: là tham số điều chỉnh kích thước đầu ra của lớp ConvTranspose2d.
            #Điều này được sử dụng để đảm bảo kích thước đầu ra phù hợp sau khi áp dụng lớp ConvTranspose2d.
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
#thực hiện tính toán sự sai khác bình phương giữa dữ liệu đầu ra và dữ liệu mục tiêu.
# Trong trường hợp này, đầu ra của autoencoder là dữ liệu giải nén và dữ liệu mục tiêu là dữ liệu gốc.
criterion = nn.MSELoss()
# tạo một trình tối ưu hóa kiểu Adam (Adam optimizer)
#https://viblo.asia/p/thuat-toan-toi-uu-adam-aWj53k8Q56m
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001 )#là tốc độ học (learning rate)

# training
for epoch in range(100):
    running_loss = 0.0
    for data in train_loader:
        #data là 1 tuple (img, nhãn)
        #nên lọc img và sử dụng mỗi img, không cần nhãn
        img, _ = data
        #ép kiểu để tensor có thể theo dõi và đưa vào CPU hoặc GPU

        img = Variable(img).to(device)
        #đặt gradient về 0 mỗi lần chạy
        optimizer.zero_grad()
        outputs = autoencoder(img)

        loss = criterion(outputs, img)
        #Tính gradient của hàm mất mát theo các tham số của mô hình để chuẩn bị cho quá trình cập nhật
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('Epoch [%d], Loss: %.4f' % (epoch+1, running_loss/len(train_loader)))

# test
#đưa vào ché độ evaluation(chế độ đánh giá)
autoencoder.eval()
test_loss = 0.0
#thường được sử dụng trong quá trình kiểm tra và đánh giá để tránh việc sử dụng bộ nhớ cho các gradient không cần thiết
with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = Variable(img).to(device)
        outputs = autoencoder(img)
        loss = criterion(outputs, img)
        #Mất mát giữa dữ liệu kiểm tra gốc và dữ liệu giải nén
        test_loss += loss.item()

print('Test Loss: %.4f' % (test_loss/len(test_loader)))

#### comparison bertween classes in test data
autoencoder.eval()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# 1 list chứa 6 list rỗng
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
    #np.mean là tính trung bình theo 1 trục cụ thể
    class_means.append(np.mean(class_encodings[i]))
    #tính bình phương sai số (độ lệch chuẩn) mean squared error
    #tham khảo thêm: https://websitehcm.com/mean-squared-error/
    class_mse = np.mean((class_encodings[i] - class_means[i])**2)
    class_mses.append(class_mse)

plt.figure(figsize=(8, 6))
plt.bar(np.arange(6)-0.2, class_means, width=0.4, label='Mean', alpha=0.5)
plt.bar(np.arange(6)+0.2, class_mses, width=0.4, label='MSE', alpha=0.5)
plt.xticks(range(6), class_names, rotation=90)
plt.legend()
plt.title("Class Encodings Mean and MSE Comparison")
plt.show()