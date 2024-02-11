import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import torchvision.models as models
import pandas as pd
import os
import torch
import json

from torchvision.transforms import transforms
from torch.utils.data import ConcatDataset


class AphidDamageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.annotations_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        image_annotations = self.annotations_frame[self.annotations_frame['filename'] == img_name]

        boxes = []
        labels = []
        for row in image_annotations.itertuples(index=False):

            region_shape_attributes = json.loads(row.region_shape_attributes)
            region_attributes = json.loads(row.region_attributes)
            # Determine the label based on the 'type' field in the region_attributes
            label_str = region_attributes['type']
            label = 1 if label_str == 'damage' else 0

            # bounding box is a square
            if 'x' in region_shape_attributes and 'y' in region_shape_attributes:
                x0 = region_shape_attributes['x']
                y0 = region_shape_attributes['y']
                x1 = x0 + region_shape_attributes['width']
                y1 = y0 + region_shape_attributes['height']
            #bounding box is a circle
            elif 'cx' in region_shape_attributes and 'cy' in region_shape_attributes:
                cx = region_shape_attributes['cx']
                cy = region_shape_attributes['cy']
                r = region_shape_attributes['r']
                x0 = cx - r
                y0 = cy - r
                x1 = cx + r
                y1 = cy + r
            else:
                raise ValueError('Unknown shape type')

            boxes.append([x0, y0, x1, y1])
            labels.append(label)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, {'boxes': boxes, 'labels': labels}



img_dir = './train'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create an instance of the dataset
aphid_dataset = AphidDamageDataset(csv_file='./train/leaves.csv', img_dir=img_dir, transform=transform)
model=torchvision.models.detection.fasterrcnn_resnet50_fpn()
numClasses = 2
inFeature = model.roi_heads.box
dataloader = DataLoader(aphid_dataset,batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in dataloader:  # Assuming you have a DataLoader called `dataloader`
        # Move inputs and labels to the appropriate device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: compute the model output
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")