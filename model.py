import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from gradcam import generate_grad_cam, overlay_grad_cam

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # If saved as state_dict
    if isinstance(checkpoint, dict):
        model = models.efficientnet_b3(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)  # Keep 4 output classes
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint  # Full model saved

    model.to(device)
    model.eval()
    
    class_names = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS", "UNKNOWN"]
    return model, class_names

def predict(model, image, class_names, device, threshold=0.6):
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100

    # If confidence is below threshold for NORMAL, PNEUMONIA, TUBERCULOSIS, mention OTHER FINDINGS
    if predicted_label in ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"] and confidence_score < threshold * 100:
        other_findings_label = "OTHER FINDINGS"
    else:
        other_findings_label = None  # No "OTHER FINDINGS" if confident

    # Generate Grad-CAM
    grad_cam_map = generate_grad_cam(model, input_tensor, predicted_class.item(), device)
    grad_cam_overlay = overlay_grad_cam(grad_cam_map, input_tensor)

    return predicted_label, confidence_score, grad_cam_overlay, other_findings_label
