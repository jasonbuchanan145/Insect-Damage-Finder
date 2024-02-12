
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch

from torchvision.transforms import transforms

import sys
class AphidDamageDataset(Dataset):
    def __init__(self,  img_dir, transform=None,debugMode=False):

        self.img_dir = img_dir
        self.transform = transform
        self.debugMode=debugMode

        self.unique_imgs = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    def __len__(self):
        return len(self.unique_imgs)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.unique_imgs[idx])
        image = Image.open(img_name).convert("RGB")

        # Since each image is now a single instance of damage, and the bounding box covers the entire image,
        # the bounding box is essentially the size of the image itself.
        # Therefore, the bounding box can be represented as [0, 0, width, height].
        width, height = image.size
        boxes = torch.tensor([[0, 0, width, height]], dtype=torch.float32)

        # Assuming every image represents damage, you might want to label every instance as 1 (or another appropriate label)
        labels = torch.tensor([1], dtype=torch.int64)  # Assuming '1' represents "damage"

        # Apply any transformations as needed
        if self.transform:
            image = self.transform(image)

        # The target dictionary now contains the full-image bounding box and the label
        target = {'boxes': boxes, 'labels': labels}

        return image, target

print(sys.version)
img_dir = './training/img/parsed'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create an instance of the dataset
aphid_dataset = AphidDamageDataset(img_dir=img_dir, transform=transform,debugMode=False)
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
