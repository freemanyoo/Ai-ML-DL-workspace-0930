from django.shortcuts import render

# Create your views here.
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from config.forms import ImageUploadForm
from .models import load_model


# CNN 모델 로드
model = load_model()

# 클래스 이름
class_names = ["Hammer", "Nipper"]

# 이미지 전처리
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# API: 이미지 업로드 및 분류
@csrf_exempt
def classify_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES["image"]
            image = Image.open(image_file)
            image = transform_image(image)

            # 모델 예측
            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()

            response_data = {
                "class": class_names[predicted_idx],
                "confidence": round(confidence * 100, 2),
            }
            return JsonResponse(response_data)

    return JsonResponse({"error": "Invalid request"}, status=400)
