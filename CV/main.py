import torch
import clip
from PIL import Image, ImageDraw
from ultralytics import YOLO  # YOLOv5 object detection library

def boolean_image_classification_with_detection(image_path, label):
    """
    Perform object detection and boolean classification for an image using YOLOv5 and CLIP.

    Args:
        image_path (str): Path to the input image.
        label (str): The label to classify the detected objects as matching or not.

    Returns:
        PIL.Image: Image with bounding boxes drawn around matching objects.
    """
    # Load the pre-trained YOLOv5 model
    yolo_model = YOLO("yolov5s.pt")  # Use YOLOv5 small model

    # Load the pre-trained CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    detections = yolo_model(image_path)  # Perform object detection

    draw = ImageDraw.Draw(image)

    # Iterate over detected objects
    for detection in detections[0].boxes:
        box = detection.xyxy[0].tolist()  # Bounding box coordinates
        cropped_image = image.crop(box)  # Crop the detected object

        # Preprocess the cropped image for CLIP
        processed_image = preprocess(cropped_image).unsqueeze(0).to(device)

        # Create text prompts
        prompts = [f"A photo of a {label}", f"Not a photo of a {label}"]
        text = clip.tokenize(prompts).to(device)

        # Perform classification with CLIP
        with torch.no_grad():
            image_features = clip_model.encode_image(processed_image)
            text_features = clip_model.encode_text(text)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Extract probabilities for "Yes" and "No"
        yes_score = similarity[0][0].item()
        no_score = similarity[0][1].item()

        # If the object matches the label, draw a bounding box
        if yes_score > no_score:
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{label}: {yes_score:.2f}", fill="red")

    return image


# Example usage
image_path = "Fried Salmon Steak.webp"
label = "lemon"
result_image = boolean_image_classification_with_detection(image_path, label)

# Save or show the result image
result_image.show()  # Display the image
result_image.save("output_image_with_boxes.jpg")  # Save the image