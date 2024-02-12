import cv2
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import torch
import json

from torchvision.transforms import transforms


class AphidDamageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None,debugMode=False):
        self.annotations_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.debugMode=debugMode

        # Create a list of unique image filenames
        self.unique_imgs = self.annotations_frame['filename'].unique()

        # Create a mapping from image filenames to unique numerical IDs
        self.img_to_id = {img: idx for idx, img in enumerate(self.unique_imgs)}

    def __len__(self):
        return len(self.unique_imgs)

    def __getitem__(self, idx):
        # Use the idx to get the unique image filename
        unique_img_name = self.unique_imgs[idx]
        img_name = os.path.join(self.img_dir, unique_img_name)
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        image_annotations = self.annotations_frame[self.annotations_frame['filename'] == unique_img_name]
        boxes = []
        labels = []
        area = []
        iscrowd = []
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
            # bounding box is a circle
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
            area.append((x1 - x0) * (y1 - y0))
            iscrowd.append(0)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = torch.tensor(area, dtype=torch.float32)
        # Use the mapping to get the unique ID for the current image
        image_id = self.img_to_id[unique_img_name]
        if self.debugMode:
            for box in boxes:
                start_point = (int(box[0]), int(box[1]))  # Top-left corner
                end_point = (int(box[2]), int(box[3]))  # Bottom-right corner
                color = (255, 0, 0)  # Red color in RGB
                thickness = 2  # Thickness of the rectangle border
                image = cv2.rectangle(image, start_point, end_point, color, thickness)
        image_pil = Image.fromarray(image)
        if self.debugMode:
            # Display the image with bounding boxes
            cv2.imshow('Image with Bounding Boxes', cv2.cvtColor(image_pil, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transform:
            image_pil = self.transform(image_pil)

        return image_pil, target


img_dir = './training/img'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create an instance of the dataset
aphid_dataset = AphidDamageDataset(csv_file='./training/leaves.csv', img_dir=img_dir, transform=transform,debugMode=False)
dataLoader = DataLoader(aphid_dataset, batch_size=20, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("pytorch does not think cuda is available, this will be very slow on cpu, grab some coffee or something")
#   exit(-1)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
for epoch in range(0, 10):
    for images, targets in dataLoader:
        images = list(img.to(device) for img in images)
        new_targets = []
        for target_dict in targets:
            new_target_dict = {}
            for key, value in target_dict.items():
                if isinstance(value, torch.Tensor):
                    new_target_dict[key] = value.to(device)
                else:
                    new_target_dict[key] = value
            new_targets.append(new_target_dict)

        targets = new_targets
        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Perform backpropagation
        losses.backward()

        # Update the weights
        optimizer.step()
        print(f"Loss: {losses.item()}")

torch.save(model.state_dict(), "model.pth")
