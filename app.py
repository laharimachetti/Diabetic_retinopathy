#!/usr/bin/env python3
# Diabetic Retinopathy Classification Web Application
# Flask-based web interface for QViT-DR model

import os
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to prevent GUI issues
import matplotlib.pyplot as plt
import pywt
from PIL import Image
from flask import Flask, request, render_template, jsonify

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define the model path
MODEL_PATH = 'C:/Users/user/Desktop/Project 4-2/qvit_dr_best_model.pth'

# Set up Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB

# --------------------- MODEL DEFINITION ---------------------

class QuantumInspiredLayer(nn.Module):
    """
    A layer that mimics quantum circuit behavior for feature transformation
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        # Apply quantum-inspired transformation
        phase = torch.sin(F.linear(x, self.theta, self.bias))
        amplitude = torch.cos(F.linear(x, self.theta, self.bias))
        
        # Combine amplitude and phase information (mimicking quantum interference)
        return amplitude + 0.1 * phase

class QViTDR(nn.Module):
    """
    Quantum-enhanced Vision Transformer for Diabetic Retinopathy
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=5,
                 embed_dim=96, depths=4, num_heads=8, drop_rate=0.1,
                 use_quantum_components=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_quantum_components = use_quantum_components
        
        # Ensure img_size is divisible by patch_size to avoid dimension issues
        if isinstance(img_size, int):
            # Round up to make divisible by patch_size
            img_size = (img_size // patch_size + (1 if img_size % patch_size != 0 else 0)) * patch_size
        
        # Stem: Initial convolutional layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Quantum-enhanced feature extraction blocks
        self.features = nn.ModuleList()
        for i in range(depths):
            block = nn.Sequential(
                nn.Conv2d(embed_dim * (2**i if i > 0 else 1), 
                          embed_dim * (2**(i+1) if i < depths-1 else 2**i),
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim * (2**(i+1) if i < depths-1 else 2**i)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.features.append(block)
        
        # Final pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Quantum-inspired layer if enabled
        self.final_dim = embed_dim * (2**(depths-1))
        if use_quantum_components:
            self.quantum_layer = QuantumInspiredLayer(self.final_dim, self.final_dim)
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.final_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Initial feature extraction
        x = self.stem(x)
        
        # Apply feature extraction blocks
        for block in self.features:
            x = block(x)
        
        # Global pooling
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Apply quantum layer if enabled
        if self.use_quantum_components:
            x = self.quantum_layer(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

# --------------------- IMAGE PREPROCESSING ---------------------

class MultiWaveletPreprocessor:
    """
    Advanced preprocessing for retinal fundus images using multi-wavelet decomposition
    """
    def __init__(self, wavelet='db2', level=2):
        self.wavelet = wavelet
        self.level = level
    
    def __call__(self, img):
        # Convert to numpy for wavelet transform
        img_np = np.array(img)
        
        # Convert RGB to grayscale for wavelet transform
        if len(img_np.shape) > 2:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        
        try:
            # Apply wavelet transform
            coeffs = pywt.wavedec2(img_gray, wavelet=self.wavelet, level=self.level)
            
            # Get approximation coefficients (LL) and detail coefficients (LH, HL, HH)
            approx, *details = coeffs
            
            # Normalize coefficients
            approx_norm = (approx - approx.min()) / (approx.max() - approx.min() + 1e-8)
            
            # Process detail coefficients for each level
            processed_details = []
            for level_details in details:
                level_processed = []
                for coeff in level_details:
                    coeff_norm = (coeff - coeff.min()) / (coeff.max() - coeff.min() + 1e-8)
                    level_processed.append(coeff_norm)
                processed_details.append(tuple(level_processed))
                
            # Reconstruct the image from processed coefficients
            reconstructed = pywt.waverec2([approx_norm, *processed_details], wavelet=self.wavelet)
            
            # Resize to original shape if needed
            if reconstructed.shape != img_gray.shape:
                reconstructed = cv2.resize(reconstructed, (img_gray.shape[1], img_gray.shape[0]))
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply((reconstructed * 255).astype(np.uint8))
            
            # Convert back to RGB
            if len(img_np.shape) > 2:
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                # Enhance each channel separately while preserving color information
                for i in range(3):
                    img_np[:,:,i] = cv2.addWeighted(img_np[:,:,i], 0.7, enhanced, 0.3, 0)
                return img_np
            else:
                return enhanced
        except Exception as e:
            print(f"Error in wavelet processing: {str(e)}")
            return img_np  # Return original image if wavelet processing fails

def preprocess_image(img_data, size=224, use_wavelet=True):
    """
    Preprocess image data with optional wavelet transform
    
    Args:
        img_data: Image data (file or bytes)
        size: Target image size (default: 224)
        use_wavelet: Whether to use wavelet transform (default: True)
        
    Returns:
        preprocessed_img: Preprocessed image tensor
        original_img: Original image for display
    """
    # Convert image data to numpy array
    if isinstance(img_data, bytes):
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        # Assume PIL Image
        img = np.array(img_data)
        if img.shape[2] == 4:  # If RGBA, convert to RGB
            img = img[:, :, :3]
    
    if img is None:
        raise ValueError("Failed to read image data")
    
    # Store original image for display
    original_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    # Convert to RGB for processing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop black borders
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (the retina)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img = img[y:y+h, x:x+w]
    except Exception as e:
        print(f"Warning: Failed to crop borders: {str(e)}")
    
    # Apply wavelet preprocessing if specified
    if use_wavelet:
        try:
            wavelet_processor = MultiWaveletPreprocessor(wavelet='db2', level=2)
            img = wavelet_processor(img)
        except Exception as e:
            print(f"Warning: Wavelet processing failed: {str(e)}")
    
    # Resize
    img = cv2.resize(img, (size, size))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    preprocessed_img = transform(img)
    
    return preprocessed_img, original_img

# --------------------- PREDICTION FUNCTIONS ---------------------

# Create and load model
def load_model(model_path=MODEL_PATH):
    """
    Load the trained model from path
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        model: Loaded model
    """
    try:
        # Create model
        model = QViTDR(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=5,
            embed_dim=96,
            depths=4,
            num_heads=8,
            drop_rate=0.1,
            use_quantum_components=True
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Load model at startup
try:
    model = load_model()
    print("Model loaded and ready for inference")
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    model = None

def predict_dr_grade(model, img_tensor):
    """
    Predict DR grade for an image
    
    Args:
        model: Trained model
        img_tensor: Preprocessed image tensor
        
    Returns:
        pred_class: Predicted class (0-4)
        pred_probs: Prediction probabilities for all classes
    """
    model.eval()
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        _, pred_class = outputs.max(1)
    
    return pred_class.item(), probs.squeeze().cpu().numpy()

def get_dr_grade_description(grade):
    """
    Get description for DR grade
    
    Args:
        grade: DR grade (0-4)
        
    Returns:
        description: Description of DR grade
    """
    descriptions = {
        0: "No DR (No visible signs of diabetic retinopathy)",
        1: "Mild NPDR (Microaneurysms only)",
        2: "Moderate NPDR (More than just microaneurysms but less than severe NPDR)",
        3: "Severe NPDR (More than 20 hemorrhages in each quadrant, venous beading, or IRMA)",
        4: "Proliferative DR (Neovascularization and/or vitreous/preretinal hemorrhage)"
    }
    
    return descriptions.get(grade, "Unknown grade")

def generate_result_visualization(img, pred_class, pred_probs):
    """
    Generate visualization of prediction results
    
    Args:
        img: Original image
        pred_class: Predicted class
        pred_probs: Prediction probabilities
        
    Returns:
        encoded_img: Base64 encoded image for HTML display
    """
    class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(img)
    ax1.set_title(f"Predicted: {class_names[pred_class]} (Grade {pred_class})")
    ax1.axis('off')
    
    # Display probabilities as bar chart
    bars = ax2.bar(class_names, pred_probs, color='skyblue')
    bars[pred_class].set_color('red')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    ax2.set_ylim(0, 1.0)
    ax2.set_title('Class Probabilities')
    ax2.set_ylabel('Probability')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    # Encode to base64 for HTML display
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return encoded_img

# --------------------- FLASK ROUTES ---------------------

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Handle image upload and classification"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Read and preprocess image
            img_data = file.read()
            img_tensor, original_img = preprocess_image(img_data, use_wavelet=True)
            
            # Make prediction
            pred_class, pred_probs = predict_dr_grade(model, img_tensor)
            
            # Get description
            description = get_dr_grade_description(pred_class)
            
            # Generate visualization
            result_img = generate_result_visualization(original_img, pred_class, pred_probs)
            
            # Prepare response
            result = {
                'grade': int(pred_class),
                'description': description,
                'probabilities': [float(p) for p in pred_probs],
                'visualization': result_img
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Failed to process file'}), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    if model is not None:
        return jsonify({'status': 'ok', 'model_loaded': True})
    else:
        return jsonify({'status': 'error', 'model_loaded': False}), 500

# Create templates directory and index.html if not exist
os.makedirs('templates', exist_ok=True)

# Write index.html template
with open('templates/index.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetic Retinopathy Classification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
            color: #333;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .upload-section {
            text-align: center;
            margin-bottom: 30px;
        }
        .file-input {
            margin-bottom: 15px;
        }
        .upload-btn {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .upload-btn:hover {
            background-color: #2980b9;
        }
        .upload-btn:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        .result-section {
            display: none;
            margin-top: 20px;
        }
        .result-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        .result-header {
            padding: 15px;
            background-color: #2c3e50;
            color: white;
            font-size: 18px;
        }
        .result-body {
            padding: 20px;
        }
        .grade-info {
            display: flex;
            margin-bottom: 20px;
        }
        .grade-value {
            font-size: 48px;
            font-weight: bold;
            margin-right: 20px;
            color: #2c3e50;
        }
        .grade-description {
            flex: 1;
        }
        .grade-description h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .grade-label {
            font-size: 14px;
            margin-bottom: 5px;
            color: #7f8c8d;
        }
        .probabilities {
            margin-top: 20px;
        }
        .probability-bar {
            height: 25px;
            margin-bottom: 10px;
            background-color: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }
        .probability-fill {
            height: 100%;
            background-color: #3498db;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-size: 14px;
            transition: width 0.6s ease-in-out;
            position: absolute;
            top: 0;
            left: 0;
        }
        .probability-label {
            position: absolute;
            left: 10px;
            z-index: 1;
            color: #333;
            font-weight: bold;
            font-size: 14px;
            display: flex;
            align-items: center;
            height: 100%;
        }
        .probability-value {
            position: absolute;
            right: 10px;
            z-index: 1;
            color: #333;
            font-weight: bold;
            font-size: 14px;
            display: flex;
            align-items: center;
            height: 100%;
        }
        .highest {
            background-color: #e74c3c;
        }
        .visualization {
            text-align: center;
            margin-top: 30px;
        }
        .visualization img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .info-section {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .grade-description-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .grade-description-table th, .grade-description-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .grade-description-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .grade-description-table tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: #7f8c8d;
            font-size: 14px;
        }
        .error-message {
            display: none;
            color: #e74c3c;
            background-color: #fadbd8;
            padding: 10px;
            border-radius: 4px;
            margin-top: 20px;
            text-align: center;
        }
        @media (max-width: 768px) {
            .grade-info {
                flex-direction: column;
            }
            .grade-value {
                margin-right: 0;
                margin-bottom: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Diabetic Retinopathy Classification</h1>
        
        <div class="upload-section">
            <h2>Upload Retinal Fundus Image</h2>
            <p>Select a retinal fundus image to classify the stage of diabetic retinopathy.</p>
            <form id="upload-form" enctype="multipart/form-data">
                <div class="file-input">
                    <input type="file" id="file-input" name="file" accept="image/*">
                </div>
                <button type="submit" class="upload-btn" id="upload-btn" disabled>Analyze Image</button>
            </form>
        </div>
        
        <div class="loading" id="loading">
            <div class="loader"></div>
            <p>Analyzing image...</p>
        </div>
        
        <div class="error-message" id="error-message"></div>
        
        <div class="result-section" id="result-section">
            <div class="result-card">
                <div class="result-header">Classification Result</div>
                <div class="result-body">
                    <div class="grade-info">
                        <div class="grade-value" id="grade-value">0</div>
                        <div class="grade-description">
                            <div class="grade-label">DR Grade</div>
                            <h3 id="grade-name">No DR</h3>
                            <p id="grade-description">No visible signs of diabetic retinopathy</p>
                        </div>
                    </div>
                    
                    <h3>Probability Distribution</h3>
                    <div class="probabilities" id="probabilities">
                        <!-- Probability bars will be inserted here -->
                    </div>
                </div>
            </div>
            
            <div class="visualization">
                <h3>Visualization</h3>
                <img id="result-image" src="" alt="Classification Result">
            </div>
        </div>
        
        <div class="info-section">
            <h2>About Diabetic Retinopathy (DR) Grading</h2>
            <p>Diabetic retinopathy is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels in the retina. The severity of diabetic retinopathy is classified according to the following scale:</p>
            
            <table class="grade-description-table">
                <tr>
                    <th>Grade</th>
                    <th>Classification</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>0</td>
                    <td>No DR</td>
                    <td>No visible signs of diabetic retinopathy</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>Mild NPDR</td>
                    <td>Microaneurysms only</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td>Moderate NPDR</td>
                    <td>More than just microaneurysms but less than severe NPDR</td>
                </tr>
                <tr>
                    <td>3</td>
                    <td>Severe NPDR</td>
                    <td>More than 20 hemorrhages in each of 4 quadrants, or venous beading in 2+ quadrants, or intraretinal microvascular abnormalities (IRMA)</td>
                </tr>
                <tr>
                    <td>4</td>
                    <td>Proliferative DR</td>
                    <td>Neovascularization and/or vitreous/preretinal hemorrhage</td>
                </tr>
            </table>
            
            <p><strong>Note:</strong> This application uses the QViT-DR (Quantum-enhanced Vision Transformer for Diabetic Retinopathy) model to classify retinal fundus images. The model has been trained on the APTOS 2019 dataset.</p>
        </div>
        
        <div class="footer">
            <p>&copy; 2023 QViT-DR Diabetic Retinopathy Classification Tool. All rights reserved.</p>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const fileInput = document.getElementById('file-input');
            const uploadBtn = document.getElementById('upload-btn');
            const uploadForm = document.getElementById('upload-form');
            const loadingSection = document.getElementById('loading');
            const resultSection = document.getElementById('result-section');
            const errorMessage = document.getElementById('error-message');
            
            // Grade display elements
            const gradeValue = document.getElementById('grade-value');
            const gradeName = document.getElementById('grade-name');
            const gradeDescription = document.getElementById('grade-description');
            const probabilitiesContainer = document.getElementById('probabilities');
            const resultImage = document.getElementById('result-image');
            
            // Grade names
            const gradeNames = [
                'No DR',
                'Mild NPDR',
                'Moderate NPDR',
                'Severe NPDR',
                'Proliferative DR'
            ];
            
            // Enable/disable upload button based on file selection
            fileInput.addEventListener('change', function() {
                uploadBtn.disabled = !fileInput.files.length;
                
                // Preview selected image if needed
                // ...
            });
            
            // Handle form submission
            uploadForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                if (!fileInput.files.length) {
                    showError('Please select an image to upload');
                    return;
                }
                
                const file = fileInput.files[0];
                
                // Validate file type
                if (!file.type.match('image.*')) {
                    showError('Please select a valid image file');
                    return;
                }
                
                // Create form data
                const formData = new FormData();
                formData.append('file', file);
                
                // Show loading, hide results and errors
                loadingSection.style.display = 'block';
                resultSection.style.display = 'none';
                errorMessage.style.display = 'none';
                
                // Submit form
                fetch('/classify', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading
                    loadingSection.style.display = 'none';
                    
                    if (data.error) {
                        showError(data.error);
                        return;
                    }
                    
                    // Display results
                    updateResults(data);
                    resultSection.style.display = 'block';
                })
                .catch(error => {
                    loadingSection.style.display = 'none';
                    showError('Error processing image: ' + error.message);
                });
            });
            
            // Update results display
            function updateResults(data) {
                // Update grade value and description
                gradeValue.textContent = data.grade;
                gradeName.textContent = gradeNames[data.grade];
                gradeDescription.textContent = data.description;
                
                // Update probability bars
                probabilitiesContainer.innerHTML = '';
                
                data.probabilities.forEach((prob, index) => {
                    const barContainer = document.createElement('div');
                    barContainer.className = 'probability-bar';
                    
                    const probLabel = document.createElement('div');
                    probLabel.className = 'probability-label';
                    probLabel.textContent = gradeNames[index];
                    
                    const probValue = document.createElement('div');
                    probValue.className = 'probability-value';
                    probValue.textContent = (prob * 100).toFixed(1) + '%';
                    
                    const probFill = document.createElement('div');
                    probFill.className = 'probability-fill' + (index === data.grade ? ' highest' : '');
                    probFill.style.width = (prob * 100) + '%';
                    
                    barContainer.appendChild(probLabel);
                    barContainer.appendChild(probValue);
                    barContainer.appendChild(probFill);
                    
                    probabilitiesContainer.appendChild(barContainer);
                });
                
                // Update visualization image
                resultImage.src = 'data:image/png;base64,' + data.visualization;
            }
            
            // Show error message
            function showError(message) {
                errorMessage.textContent = message;
                errorMessage.style.display = 'block';
            }
        });
    </script>
</body>
</html>''')

# Run app if executed directly
if __name__ == '__main__':
    # Check if model was loaded successfully
    if model is None:
        print("WARNING: Model could not be loaded. Application will not function correctly.")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
