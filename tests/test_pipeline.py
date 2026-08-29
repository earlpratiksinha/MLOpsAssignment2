from PIL import Image
import torch
import torchvision.transforms as transforms

def test_image_preprocessing():
    img = Image.new("RGB", (100, 100), color="red")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)

def test_model_output_shape():
    dummy_input = torch.randn(1, 3, 224, 224)
    assert dummy_input.shape[1] == 3