import torch
import cv2
import numpy as np
import torch.nn.functional as F

def generate_grad_cam(model, input_image, target_class, device):
    def hook_fn(module, input, output):
        global feature_map
        feature_map = output

    # Attach hook to the last convolutional block (EfficientNet B3)
    final_conv_layer = list(model.features.children())[-1]
    hook = final_conv_layer.register_forward_hook(hook_fn)

    # Forward pass
    model.eval()
    input_image = input_image.to(device)
    output = model(input_image)

    # Compute gradients
    class_score = output[0, target_class]
    model.zero_grad()
    class_score.backward(retain_graph=True)

    # Extract gradients
    gradients = torch.autograd.grad(class_score, feature_map)[0]
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

    # Generate Grad-CAM map
    grad_cam_map = torch.sum(weights * feature_map, dim=1, keepdim=True)
    grad_cam_map = F.relu(grad_cam_map).squeeze().cpu().detach().numpy()

    # Normalize and resize
    grad_cam_map = np.maximum(grad_cam_map, 0)
    grad_cam_map = grad_cam_map / grad_cam_map.max()
    grad_cam_map = cv2.resize(grad_cam_map, (input_image.shape[2], input_image.shape[3]))

    # Remove hook
    hook.remove()

    return grad_cam_map

def overlay_grad_cam(grad_cam_map, original_image):
    # Convert to numpy and normalize
    original_image = original_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    original_image = np.clip(original_image, 0, 1)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0

    # Overlay heatmap
    overlay = 0.4 * heatmap + 0.6 * original_image
    return overlay
