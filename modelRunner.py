import torchvision
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
image_path = './testing/img.png'
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transformed_image = transform(image)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn().cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()
with torch.no_grad():
    prediction = model([transformed_image.to("cuda")])

pred_boxes = prediction[0]['boxes'].cpu().numpy()
pred_labels = prediction[0]['labels'].cpu().numpy()
pred_scores = prediction[0]['scores'].cpu().numpy()

threshold = 0.6
filtered_boxes = pred_boxes[pred_scores >= threshold]
filtered_labels = pred_labels[pred_scores >= threshold]
filtered_scores = pred_scores[pred_scores >= threshold]

#drawing stuff
pil_image = transforms.ToPILImage()(transformed_image.cuda())

fig, ax = plt.subplots(1)
ax.imshow(image)

for box, score in zip(filtered_boxes,filtered_scores):
    x, y, x2, y2 = box
    rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, f'{score:.2f}', fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))
plt.show()