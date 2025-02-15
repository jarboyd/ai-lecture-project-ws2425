import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# Receives a list of images and targets and returns a tuple (list_images, list_targets).
def collate_fn(batch):
    return tuple(zip(*batch))


# Bounding boxes need to be resized along with the images
class ResizeWithBoxes:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        original_width, original_height = image.size
        new_height, new_width = self.size

        image = F.resize(image, self.size)

        # Load the bounding boxes and calc the scaling factor we need to apply
        boxes = target["boxes"]
        scale_w = new_width / original_width
        scale_h = new_height / original_height

        # Scale the boxes to match the new image size
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w
        scaled_boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
        target["boxes"] = scaled_boxes

        target["area"] = (scaled_boxes[:, 2] - scaled_boxes[:, 0]) * (scaled_boxes[:, 3] - scaled_boxes[:, 1])
        return image, target


# Resizes the image and applies augmentations for training
def transform_train(image, target):
    image, target = ResizeWithBoxes((600, 600))(image, target)
    # Apply color jitter for variety learning
    image = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2)(image)
    image = F.to_tensor(image)
    return image, target


# Class to hold the specify data from the dataset
class TrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.image_bboxes = {} # Image file name to bounding boxes
        self.transforms = transforms

        with open(annotation_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: image_file.ppm;left;top;right;bottom;ClassID but we don't need ClassID
                img_name, left, top, right, bottom, class_id = line.split(";")
                left, top, right, bottom = map(int, [left, top, right, bottom])
                if img_name not in self.image_bboxes:
                    self.image_bboxes[img_name] = []
                # Only one label since we want to only detect if a traffic sign is there, not what kind of sign
                self.image_bboxes[img_name].append({"bbox": [left, top, right, bottom], "label": 1})

        self.img_files = list(self.image_bboxes.keys())

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        boxes, labels = [], [] # One image can have multiple bounding boxes
        for ann in self.image_bboxes[img_file]:
            boxes.append(ann["bbox"])
            labels.append(ann["label"])

        # Convert lists to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms: # To apply transforms like color changes fro training
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)

        return img, target


# Loads a pre-trained Faster R-CNN model
# The head is replaced by our own with the number of classes we need (tarffic sign and background)
def get_faster_rcnn_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Class to train the model
# 10 epochs were enough for a good result, 20 only slightly improved it, 12 were a good tradeoff between time and result
class TrafficSignTrainer:
    def __init__(self, images_dir, annotation_file, model_save_path, batch_size=4, num_workers=2, lr=0.005, num_epochs=12, device=None):
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.num_epochs = num_epochs

        # If possible use CUDA, so much faster
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Create the full dataset and assign the training transform to alter the images
        complete_dataset = TrafficSignDataset(images_dir, annotation_file, transforms=transform_train)

        # We could split into train and validation here, but don't really need it        
        self.train_dataset = complete_dataset

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

        # Create the model and move it to the target device
        self.model = get_faster_rcnn_model(num_classes=2)  # background and traffic sign are thr two classes, we don't do classification here
        self.model.to(self.device)
        self.model.train()

        # Set up the optimizer and mixed precision scaler
        self.optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=self.lr, momentum=0.9, weight_decay=0.0005)
        self.scaler = torch.amp.GradScaler("cuda")

    def train(self):
        print(f"Starting training on device: {self.device}")
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_i, (images, targets) in enumerate(self.train_loader):
                # Move images and targets to the specified device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Use mixed precision is potentially faster on GPU
                with torch.amp.autocast("cuda"):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                # Backpropagation steps
                self.optimizer.zero_grad()
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += losses.item()
                if (batch_i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_i+1}], Loss: {losses.item():.4f}")
            
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] Average Loss: {avg_loss:.4f}")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

"""
if __name__ == "__main__":
    images_dir = "C:/Users/malte/Desktop/ki/data/"
    annotation_file = "C:/Users/malte/Desktop/ki/gt.txt"
    model_save_path = "C:/Users/malte/Desktop/ki/faster_rcnn_traffic_signs_final.pth"

    trainer = TrafficSignTrainer(images_dir, annotation_file, model_save_path, batch_size=4, num_workers=2, lr=0.005, num_epochs=12)
    trainer.train()
    trainer.save_model()
"""