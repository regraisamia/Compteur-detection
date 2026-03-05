# Meter Digit Detection Project

## Overview
This project implements an automated meter reading system using deep learning. It combines a custom-trained CNN model with OCR capabilities to detect and read digits from utility meter images.

## Project Structure

```
projet/
├── modele.ipynb              # Model training notebook
├── app3.py                   # Streamlit application
├── Model_V2.tflite          # Trained TFLite model
├── meterdigits/             # Training dataset (100 classes: 0.0-9.9)
├── test_images/             # Test images for validation
```

## Features

### 1. Model Training (`modele.ipynb`)
- **Architecture**: Custom CNN with 3 convolutional blocks
  - Input: 32x20x3 RGB images
  - Conv blocks with LeakyReLU activation and BatchNormalization
  - GlobalAveragePooling2D for robustness
  - Output: 10 classes (digits 0-9)
  
- **Dataset**:
  - Combined dataset from two sources (18,890 images total)
  - Filtered transition states (0.3-0.7) for clearer predictions
  - Data augmentation (rotation, shift, zoom, shear)
  
- **Training**:
  - Adam optimizer (lr=0.001)
  - Categorical crossentropy loss
  - EarlyStopping and ReduceLROnPlateau callbacks
  - Final accuracy: ~97.5%

- **Model Export**:
  - Quantized TFLite format for efficient deployment
  - INT8 quantization with float32 I/O

### 2. Application (`app3.py`)
A Streamlit-based web application with:

#### Automatic Meter Detection
- Uses EasyOCR for initial digit zone detection
- Image inversion technique for better OCR performance
- Smart filtering:
  - Rejects white/bright backgrounds
  - Keeps dark backgrounds (black/red meter displays)
  - Merges nearby detection boxes
  - Selects best candidate based on aspect ratio

#### Digit Recognition
- Segments detected meter zone into individual digits
- Preprocesses each digit (resize to 20x32, normalize)
- Uses TFLite model for prediction
- Displays confidence scores

#### User Interface
- Image upload functionality
- Adjustable parameters:
  - Number of digits (4-10)
  - Edge cropping margin
- Visual feedback:
  - Original image display
  - Detection boxes (green=accepted, red=rejected)
  - Individual digit crops with predictions
  - Formatted meter reading output

## Installation

### Requirements
```bash
pip install streamlit tensorflow pillow easyocr opencv-python numpy matplotlib scikit-learn
```

### Dataset Setup
1. Extract `meterdigits.zip` to `dataset/meterdigits/`
2. Extract `dataYahya.zip` to `dataset/newDataset/`

## Usage

### Training the Model
```bash
# Open and run modele.ipynb in Jupyter/Colab
# The notebook will:
# 1. Load and preprocess datasets
# 2. Train the CNN model
# 3. Export to Model_V2.tflite
```

### Running the Application
```bash
streamlit run app3.py
```

Then:
1. Upload a meter image
2. Adjust parameters if needed
3. Click "Lancer l'analyse"
4. View results and individual digit predictions

## Model Performance

### Training Results
- Training accuracy: ~97.8%
- Validation accuracy: ~97.5%
- Model size: ~158K parameters (617KB)
- Quantized size: Significantly reduced for deployment

### Key Improvements
- Filtered ambiguous transition states (0.3-0.7)
- Combined multiple datasets for better generalization
- Data augmentation for robustness
- Batch normalization for stable training

## Technical Details

### Image Preprocessing
```python
- Resize: 20x32 pixels
- Color: RGB (3 channels)
- Normalization: [0, 1] range
```

### Detection Strategy
1. Invert image for better OCR
2. Detect all digit zones
3. Filter by background brightness
4. Merge aligned boxes
5. Select best candidate (aspect ratio > 2.0)

### Prediction Pipeline
1. Crop meter zone
2. Segment into N digits
3. Preprocess each digit
4. Run TFLite inference
5. Format output (e.g., "12,345")

## Limitations & Future Work

### Current Limitations
- Requires relatively clear meter images
- Fixed input size (20x32)
- Limited to single-line meter displays

### Potential Improvements
- Multi-scale detection
- Support for different meter types
- Real-time video processing
- Mobile deployment optimization
- Confidence threshold tuning

## Files Description

### Core Files
- `modele.ipynb`: Complete training pipeline
- `app3.py`: Production application
- `Model_V2.tflite`: Quantized inference model

### Data Files
- `meterdigits/`: Main training dataset (100 classes)
- `test_images/`: Validation images
- `dataYahya.zip`: Additional training data

## Authors
Samia REGRAI

## Acknowledgments
- Dataset sources: meterdigits and dataYahya
- EasyOCR for text detection
- TensorFlow/Keras for model training
- Streamlit for web interface
