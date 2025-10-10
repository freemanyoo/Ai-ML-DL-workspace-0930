# Create your views here.
import torch
import torchvision.transforms as transforms
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import serializers
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from rest_framework.routers import DefaultRouter
from rest_framework.views import APIView

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

# 레스트풀 형식의 컨트롤러 ,
class ImageClassificationAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            from rest_framework.response import Response
            from rest_framework import status
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        image_file = request.FILES['image']
        if not isinstance(image_file, InMemoryUploadedFile):
            return Response({'error': 'Invalid file type'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image = Image.open(image_file).convert('RGB')
            image = transform_image(image)

            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()

            response_data = {
                'class': class_names[predicted_idx],
                'confidence': round(confidence * 100, 2)
            }
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Serializer 정의
class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()

# 라우터 등록
router = DefaultRouter()


class ImageClassificationViewSet:
    pass


router.register(r'classify-image', ImageClassificationViewSet, basename='classify-image')
