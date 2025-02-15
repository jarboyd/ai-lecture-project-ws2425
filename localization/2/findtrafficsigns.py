import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

def get_faster_rcnn_model(num_classes=2):
    # Load a pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the head with our required number of classes (2, signs and not signs (background))
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

class TrafficSignDetector:
    def __init__(self, model_path, target_size=(600, 600), num_classes=2, device=None):
        # Allow to pass device, so we can use shared device with the classification network
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Target size, format is HEIGHT, WIDTH !!! (because torchvision uses that)
        self.target_size = target_size
        
        # Load the model
        self.model = get_faster_rcnn_model(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # We want to evaluate
        
        # HEIGHT, WIDTH !!! is expected
        self.transform = T.Compose([T.Resize((self.target_size[0], self.target_size[1])), T.ToTensor()])
        
        # Only values above this are detected, 0.5 works quite well
        self.detection_threshold = 0.5

    # Prefer numpy array, but normal images work too and are then converted
    def predict_boxes(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Just in case, convert color
        elif not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL Image or a numpy array.")
        
        orig_width, orig_height = image.size  # PIL returns (width, height)
        
        # Will be resized to target_size
        image_resized = self.transform(image)  # Tensor of shape [C, H, W]
        
        # Run inference
        with torch.no_grad():
            # The model expects a list of images, so we give it a list of one
            prediction = self.model([image_resized.to(self.device)])[0]
        
        # Predicted boxes and the confidence score
        pred_boxes = prediction["boxes"].cpu().numpy() # The boxes are in the resized coordinates
        pred_scores = prediction["scores"].cpu().numpy()
        
        # Drop everything below the threshold
        valid_boxes = []
        for box, score in zip(pred_boxes, pred_scores):
            if score >= self.detection_threshold:
                valid_boxes.append(box)
        
        # Scales for converting the coordinates back
        target_width = self.target_size[1]  # target_size is (height, width)
        target_height = self.target_size[0]
        scale_x = orig_width / target_width
        scale_y = orig_height / target_height
        
        # Convert the scales back to the original image size
        mapped_boxes = []
        for box in valid_boxes:
            x1, y1, x2, y2 = box

            x1_orig = x1 * scale_x
            y1_orig = y1 * scale_y
            x2_orig = x2 * scale_x
            y2_orig = y2 * scale_y
            mapped_boxes.append((x1_orig, y1_orig, x2_orig, y2_orig))
        
        return mapped_boxes

    def crop_boxes(self, image):
        # Predicted bounding boxes in the original image size
        boxes = self.predict_boxes(image)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            raise ValueError("Input image must be a PIL Image or a numpy array.")
        
        cropped_images = []
        for box in boxes:
            # Image has pixel coordinates, int
            x1, y1, x2, y2 = map(int, box)
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_images.append(cropped_img)
        
        return cropped_images

"""
if __name__ == "__main__":
    MODEL_PATH = "C:/Users/malte/Desktop/ki/faster_rcnn_traffic_signs_final.pth"
    
    detector = TrafficSignDetector(model_path=MODEL_PATH)
    test_image_path = "C:/Users/malte/Desktop/ki/bild_autobahn_schilder_tempo.jpg.webp"
    image = Image.open(test_image_path).convert("RGB")
    boxes = detector.predict_boxes(image)
    print("Predicted bounding boxes:")
    for box in boxes:
        print(box)
    cropped_imgs = detector.crop_boxes(image)
    for idx, crop in enumerate(cropped_imgs):
        crop.show(title=f"Cropped Image {idx+1}")
"""