import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

# Download the pretrained model
model = models.resnet34(pretrained=True, progress=True)
model.eval()

# The 1000 categories of images: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

# function to transform image
data = torch.rand((5, 3, 224, 224))
pred = model(data)

print(pred.size())
probs, indices = pred.topk(5, dim=1)
print(probs)
print(indices)

# Show the resized and cropped image
img1 = torch.unsqueeze(transforms.ToTensor()(Image.open('./img/mimi1.jpg')), 0)
pred1 = model(img1)
percents = torch.nn.functional.softmax(pred1, dim=1)[0] * 100
probs, indices = percents.topk(5)
print(probs)
print(indices)

# Load an actual image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])
img = Image.open('./img/mimi2.jpg')
img = transform(img)
img = torch.unsqueeze(img, 0)
pred = model(img)

print(pred.size())
percents = torch.nn.functional.softmax(pred, dim=1)[0] * 100
probs, indices = percents.topk(5)
print(probs)
print(indices)
