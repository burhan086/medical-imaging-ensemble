import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import urllib.request

# --- UI Setup ---
st.set_page_config(page_title="Medical Image Diagnosis", layout="wide")
st.title("Multi-Model Medical Image Diagnosis System")
st.write("Upload a Chest X-Ray to receive an ensemble diagnosis (ResNet50 + ViT) and Grad-CAM analysis.")

# --- Auto-Download Model Weights ---
@st.cache_resource
def download_weights():
    # 👇 REPLACE THESE WITH YOUR ACTUAL GITHUB RELEASE LINKS 👇
    resnet_url = "https://github.com/burhan086/medical-imaging-ensemble/releases/download/v1.0/resnet50_pneumonia.pth"
    vit_url = "https://github.com/burhan086/medical-imaging-ensemble/releases/download/v1.0/vit_pneumonia.pth"
    
    if not os.path.exists('resnet50_pneumonia.pth'):
        with st.spinner("Downloading ResNet50 weights (~90MB)... Please wait."):
            urllib.request.urlretrieve(resnet_url, 'resnet50_pneumonia.pth')
            
    if not os.path.exists('vit_pneumonia.pth'):
        with st.spinner("Downloading ViT weights (~330MB)... This may take a minute."):
            urllib.request.urlretrieve(vit_url, 'vit_pneumonia.pth')

# Call the download function before loading models
download_weights()

# --- Load Models (Cached so they don't reload on every click) ---
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ResNet
    resnet = models.resnet50()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 2))
    resnet.load_state_dict(torch.load('resnet50_pneumonia.pth', map_location=device))
    resnet = resnet.to(device)
    resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = True # Required for Grad-CAM
        
    # Load ViT
    vit = models.vit_b_16()
    num_ftrs_vit = vit.heads.head.in_features
    vit.heads.head = nn.Linear(num_ftrs_vit, 2)
    vit.load_state_dict(torch.load('vit_pneumonia.pth', map_location=device))
    vit = vit.to(device)
    vit.eval()
    
    return resnet, vit, device

resnet, vit, device = load_models()

# --- Image Processing ---
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original X-Ray")
        st.image(image, use_container_width=True)
        
    # Run Inference
    with st.spinner("Analyzing with Ensemble Model..."):
        input_tensor = test_transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob_resnet = F.softmax(resnet(input_tensor), dim=1)
            prob_vit = F.softmax(vit(input_tensor), dim=1)
            ensemble_prob = (prob_resnet + prob_vit) / 2.0
            confidence, predicted_class = torch.max(ensemble_prob, 1)
            
        classes = ['Normal', 'Pneumonia']
        result = classes[predicted_class.item()]
        conf_score = confidence.item() * 100
        
    st.success(f"**Diagnosis:** {result} (Confidence: {conf_score:.2f}%)")
    
    # Generate Grad-CAM
    with st.spinner("Generating Explainability Heatmap..."):
        target_layers = [resnet.layer4[-1]]
        cam = GradCAM(model=resnet, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_class.item())]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        img_resized = image.resize((224, 224))
        rgb_img = np.float32(img_resized) / 255 
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
    with col2:
        st.subheader("Grad-CAM Focus Area")
        st.image(visualization, use_container_width=True)
