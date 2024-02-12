import torchvision
from PIL import Image
import torch
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Load the image
image_path = './testing/leaf1.jpg'
image = Image.open(image_path).convert("RGB")

# Transform the image
# Note: adjust the transform to match your training preprocessing
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transformed_image = transform(image)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Turn off gradient computation
    prediction = model([transformed_image.to("cpu")])  # Assuming 'device' is your computation device (CPU or GPU)

# Process predictions
# The prediction is a list of dictionaries, one for each input image
# Here, we're processing just one image, so we take the first element
pred_boxes = prediction[0]['boxes'].cpu().numpy()  # Bounding boxes
pred_labels = prediction[0]['labels'].cpu().numpy()  # Labels (damage or not)
pred_scores = prediction[0]['scores'].cpu().numpy()  # Confidence scores

# Filter predictions based on a confidence threshold (e.g., 0.5)
threshold = 0.1
filtered_boxes = pred_boxes[pred_scores >= threshold]
filtered_labels = pred_labels[pred_scores >= threshold]
filtered_scores = pred_scores[pred_scores >= threshold]

# Now, 'filtered_boxes', 'filtered_labels', and 'filtered_scores' contain the predictions you're interested in
# You can use this data to visualize the predictions on the image, analyze the detected damage, etc.
pil_image = T.ToPILImage()(transformed_image.cpu())

# Create a figure and axis for plotting
fig, ax = plt.subplots(1)
ax.imshow(pil_image)

# Draw each bounding box
for box in filtered_boxes:
    x, y, x2, y2 = box
    rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()