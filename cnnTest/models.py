# Create your models here.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


# CNN 모델 정의 (Hammer, Nipper 분류)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 로드 함수
def load_model():
    import os
    from config import settings
    model_path = os.path.join(settings.BASE_DIR, "custom_cnn_251002.pth")
    model = CustomCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 교체1
# ResNet50 모델 로드 함수
# import torchvision.models as models, 수동 임포트
def load_resnet50():
    import os
    from config import settings
    model_path = os.path.join(settings.BASE_DIR, "resnet50-251010_model.pth")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Hammer / Nipper 분류 (2개 클래스)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model