import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import MaskRCNN
from torchvision.transforms import functional as F
from torchvision.ops import box_iou
from torch.utils.data import Dataset, DataLoader
import os
import json
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self, root_dir, subset):
        self.root_dir = root_dir
        self.subset = subset
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.image_dir = os.path.join(root_dir, subset)
        self.image_files = [file for file in os.listdir(self.image_dir) if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]
        self.image_annotation_pairs = self._load_annotations()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        filename = self.image_files[idx]
        annotation = None
        for i in range(len(self.image_annotation_pairs[0])+1):
            if self.image_annotation_pairs[0][i]['file_name'] == filename:
                # print(self.image_annotation_pairs[0][i]['file_name'], self.image_annotation_pairs[0][i]['id'])
                annotation = self.image_annotation_pairs[1][i]
                # print(self.image_annotation_pairs[0][i]['file_name'],self.image_annotation_pairs[0][i]['id'], annotation)
                break

        segmentations = []
        boxes = []
        labels = []
        for key, value in annotation.items():
            if key == 'category_id':
                labels.append(value)
            elif key == 'bbox':
                x_min, y_min, width, height = value
                x_max = x_min + width
                y_max = y_min + height
                boxes.append(x_min)
                boxes.append(y_min)
                boxes.append(x_max)
                boxes.append( y_max)
            elif key == 'segmentation':
                for v in value:
                    segmentations = v

        masks = self.create_mask(segmentations, image.size)

        boxes = torch.as_tensor([boxes] , dtype = torch.float32)
        labels = torch.ones((labels[0],) , dtype = torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks

        return self.transforms(image) , target

    def _load_annotations(self):
        annotation_file = os.path.join(self.image_dir, '_annotations.coco.json')
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations['images'], annotations['annotations']

    def create_mask(self, segmentation, size):
        masks=[]
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(segmentation, outline=1, fill=1)
        # mask_array = np.array(mask)
        # plt.imshow(mask_array, cmap='gray')
        # plt.show()
        mask_array = np.array(mask)
        mask_array = np.expand_dims(mask_array, axis=0)
        masks.append(torch.as_tensor(mask_array, dtype=torch.uint8))
        return torch.as_tensor(mask_array, dtype=torch.uint8)

def train_model(model, optimizer, train_loader, val_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    bbox_ious = []
    mask_ious = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_bbox_iou = 0.0
        epoch_mask_iou = 0.0
        total_batches = 0
        for images, targets in train_loader:
            images_new = [image.to(device) for image in images]
            targets_new = []

            for i in range(len(images)):
                single_dict = {
                    'boxes': targets['boxes'][i].to(device),
                    'labels': targets['labels'][i].to(device),
                    'masks': targets['masks'][i].to(device)
                }
                targets_new.append(single_dict)

            loss_dict = model(images_new, targets_new)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss}")
        iou, mask_iou = validate_model(model, val_loader, device)
        bbox_ious.append(iou)
        mask_ious.append(mask_iou)
    print("Training complete")
    return train_losses, bbox_ious, mask_ious

def calculate_iou(pred_boxes, target_boxes):
    ious = box_iou(pred_boxes, target_boxes)
    return ious.mean().item()

def calculate_mask_iou(pred_mask, target_mask):
    intersection = (pred_mask * target_mask).sum().item()
    union = (pred_mask + target_mask).clamp(0, 1).sum().item()
    iou = intersection / union if union > 0 else 0.0
    return iou

    return total_loss
def validate_model(model, val_loader, device):
    model.eval()
    total_iou_bbox = 0.0
    total_iou_mask = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]
            targets=[targets]
            
            predictions = model(images)
            for pred, target in zip(predictions, targets):
                pred_boxes = pred['boxes'].to(device)
                target_boxes = target['boxes'][0].to(device)

                #iou for bounding boxes
                iou_bbox = calculate_iou(pred_boxes, target_boxes)
                total_iou_bbox += iou_bbox

                #iou for masks
                pred_masks = pred['masks'].to(device)
                target_masks = target['masks'][0].to(device)

                pred_masks_binary = (pred_masks > 0.5).float()
                target_masks_binary = (target_masks > 0.5).float()

                iou_mask = calculate_mask_iou(pred_masks_binary, target_masks_binary)
                total_iou_mask += iou_mask

                total_batches += 1

    avg_iou_bbox = total_iou_bbox / total_batches if total_batches > 0 else 0.0
    avg_iou_mask = total_iou_mask / total_batches if total_batches > 0 else 0.0

    print(f"Average IoU (Bounding Boxes): {avg_iou_bbox}")
    print(f"Average IoU (Masks): {avg_iou_mask}")
    return avg_iou_bbox, avg_iou_mask

def plot_losses(train_losses, bbox_ious, mask_ious):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label="Training Loss")

    plt.plot(epochs, bbox_ious, label="Bounding Box IoU")
    plt.plot(epochs, mask_ious, label="Mask IoU")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training and Validation Loss + IoU Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_prediction_mask(model, image):
    device = next(model.parameters()).device
    transform = transforms.Compose([transforms.ToTensor()])
    input_image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(input_image)
    pred_mask = predictions[0]['masks'][0, 0]
    return pred_mask.cpu().numpy()

def mask_to_coordinates(mask):
    binary_mask = (mask > 0.5).astype(np.uint8)
    mask_image = Image.fromarray(binary_mask * 255)
    mask_array = np.array(mask_image)
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for contour in contours:
        contour = np.squeeze(contour, axis=1)
        coordinates.append(contour.tolist())
    return coordinates

def main():
    train_dataset = CustomDataset(root_dir="/content/", subset="train")
    val_dataset = CustomDataset(root_dir="/content/", subset="val")
    test_dataset = CustomDataset(root_dir="/content/", subset="test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)

    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features , 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask , hidden_layer , 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    train_losses, bbox_iou, mask_iou = train_model(model, optimizer, train_loader, val_loader, num_epochs=100)
    plot_losses(train_losses, bbox_iou, mask_iou)

    # rgb_image = cv2.imread("/content/MyImages/short.png")
    # rgb_image = cv2.imread("/content/MyImages/long.png")
    rgb_image = cv2.imread("/content/MyImages/door.png")
    rgb_image = cv2.resize(rgb_image, (550, 412))
    prediction_mask = generate_prediction_mask(model, rgb_image)
    plt.imshow(prediction_mask, cmap='gray')
    plt.show()
    coordinates = mask_to_coordinates(prediction_mask)
    print("Coordinates of the mask:")
    for coord in coordinates:
        print(coord)

if __name__ == "__main__":
    main()
