import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path, device):
    # Load trained EfficientNet model
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Define class names
    class_names = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]
    
    return model, class_names

def predict(model, image, class_names, device):
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100

    # Generate Grad-CAM
    from gradcam import generate_grad_cam, overlay_grad_cam
    grad_cam_map = generate_grad_cam(model, input_tensor, predicted_class.item(), device)
    grad_cam_overlay = overlay_grad_cam(grad_cam_map, input_tensor)

    return predicted_label, confidence_score, grad_cam_overlay
