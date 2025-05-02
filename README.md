# Whisky Goggles

Whisky Goggles is a Streamlit-based web application developed for BAXUS to scan whisky bottle labels from images and match them to a database of 500 whisky bottles. Using computer vision (OpenCV) and OCR (PaddleOCR), it detects and recognizes labels, extracts text, and identifies bottles with high accuracy, even under varying lighting, angles, and partial labels. The app outputs color images with detected labels and supports a database search by name, spirit type, and price range for recording pricing information at liquor stores.

## Overview

The goal of Whisky Goggles is to develop a computer vision system that enables users to:

- Scan whisky bottle labels from photos and display results in color.
- Match identified labels to BAXUS's dataset of 500 bottles with robust text and feature validation.
- Provide structured output with confidence scores for matches.
- Facilitate pricing data collection in liquor stores via a searchable database.

## Features

### Label Detection & Recognition:
- Processes images of whisky bottle labels using PaddleOCR with angle classification for rotated text.
- Extracts visual features using ORB (Oriented FAST and Rotated BRIEF) and text with high accuracy.
- Matches labels to the BAXUS dataset using keyword-based text filtering and fuzzy matching.
- Displays detected label regions in color with green bounding boxes.

### Bottle Identification:
- Achieves high accuracy for the 500-bottle dataset, handling variations in lighting, angles, and partial labels.
- Combines feature matching (60%) and text matching (40%) for confidence scores.
- Displays low-confidence matches with warnings and handles cases with limited text via relaxed validation.
- Outputs structured data including bottle details, confidence score, and extracted text.

### Database Search:
- Search whiskeys by name, spirit type, and price range (MSRP, fair price, shelf price).
- Presents results in a clean, tabular format using Streamlit's dataframe.

### User Interface:
- Built with Streamlit for an intuitive, responsive UI.
- Supports image uploads, a clear button, and styled output with custom CSS and Google Fonts (Playfair Display).

### Performance:
- Uses parallel processing (ThreadPoolExecutor) for multiple label regions.
- Stores preprocessed ORB descriptors in HDF5 for efficient matching.
- Sorts results by confidence for better usability.

### Noise Reduction:
- Filters OCR output to remove non-alphabetic characters and retain words ≥3 characters.
- Requires high-confidence OCR results (≥0.8) but includes fallback for low-text scenarios.

## Technical Requirements

### Core Functionality

#### Label Detection & Recognition:
- Detects whisky bottle labels in color images using PaddleOCR.
- Extracts ORB visual features and text, matching against the BAXUS 500-bottle dataset.

#### Bottle Identification:
- Achieves robust identification with handling of lighting, angle, and partial label variations.
- Outputs structured data with confidence scores and color images.

### Technical Specifications

#### Computer Vision Approach:
- Feature Matching: ORB for visual feature extraction, matched using BFMatcher with Hamming distance and homography (RANSAC).
- OCR: PaddleOCR with angle classification, quad bounding boxes, and slow detection mode for accuracy.
- Text Matching: Keyword-based scoring (≥1 matching keyword) and fuzzy matching (≥80% similarity, or ≥90% without keywords).

#### Output Format:
- JSON-like structure with bottle details (name, spirit type, prices, score), confidence score, extracted text, and color sub-image.

## Implementation Details

### Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Image Processing | opencv-python-headless | ORB feature detection, preprocessing |
| OCR | paddleocr | Label detection, text extraction |
| Data Handling | numpy, pandas, h5py | CSV/HDF5 processing, data manipulation |
| Web Framework | streamlit | User interface and app deployment |
| Image Downloading | requests | Fetch images from dataset URLs |
| Text Matching | fuzzywuzzy, python-Levenshtein | Fuzzy text matching |
| Parallel Processing | concurrent.futures | Process multiple label regions |
| System Dependencies | libgl1-mesa-glx, libglib2.0-0 | OpenCV and PaddleOCR support |

### Dataset:
- `wine.csv`: Contains 500 whiskey records with columns: id, name, spirit_type, image_url, avg_msrp, fair_price, shelf_price, total_score.
- Preprocessed images stored in `dataset/`, ORB descriptors and keypoints in `reference_data.h5`.

## Workflow

### Preprocessing:
- Downloads images from image_url in wine.csv using ProcessPoolExecutor.
- Resizes images to 320x240 and normalizes using CLAHE for feature detection.
- Extracts ORB keypoints and descriptors, stored in reference_data.h5.

### Label Detection:
- Uses PaddleOCR to detect text regions in uploaded color images (max dimension 1000 pixels).
- Filters regions by area (≥0.005 of image area), aspect ratio (0.3–3.0), and size (width >30, height >20).
- Draws green bounding boxes on color images, limiting to MAX_REGIONS=5.

### Text Extraction and Matching:
- Extracts text using PaddleOCR, requiring ≥0.8 confidence and ≥3 characters.
- Cleans text by removing non-alphabetic characters and retaining words ≥3 characters.
- Matches text against whiskey names using:
  - Keyword-based scoring (≥1 common keyword from a set including 'whisky', 'bourbon', etc.).
  - Fuzzy matching (≥80% similarity with keywords, ≥90% without).
  - Fallback validation for limited text to reduce skipped validations.

### Bottle Identification:
- Computes ORB descriptors for label regions (grayscale for feature detection).
- Matches descriptors against the dataset using BFMatcher with homography.
- Combines feature confidence (based on inlier matches) and text confidence (1.0 for match, 0.5 for OCR failure, 0.0 for mismatch).
- Outputs matches with confidence ≥0.7 as "success" and lower as "low_confidence" with warnings.
- Returns color sub-images for display.

### Search Functionality:
- Filters wine.csv by name, spirit type, and price range.
- Displays results in a Streamlit dataframe.

### Parallel Processing:
- Processes multiple label regions concurrently using ThreadPoolExecutor (max 4 workers).
- Sorts results by confidence for user-friendly display.

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| THRESHOLD | 30 | Min good feature matches for identification |
| MIN_KEYPOINTS | 50 | Min keypoints for feature matching |
| MIN_TEXT_SIMILARITY | 80 | Min fuzzy matching score (%) |
| IMAGE_SIZE | (320, 240) | Standardized image resolution |
| MAX_REGIONS | 5 | Max label regions to process |
| CONFIDENCE_THRESHOLD | 0.7 | Min combined confidence for "success" matches |
| OCR_CONFIDENCE_THRESHOLD | 0.8 | Min PaddleOCR confidence for text |

## Prerequisites

- Python: 3.12 (tested with current script).
- System Dependencies (local development):
  - Linux: libgl1-mesa-glx, libglib2.0-0.
- Streamlit Cloud: Configured via packages.txt.

## Setup Instructions

### Local Development

1. **Clone the Repository**:
   ```
   git clone https://github.com/Omerhrr/Whiskey_Googles.git
   cd Whiskey_Googles
   ```

2. **Install System Dependencies (Linux)**:
   ```
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
   ```

3. **Install Python Dependencies**:
   ```
   pip install -r requirements.txt
   ```

   Contents of requirements.txt:
   ```
   opencv-python-headless==4.10.0.84
   numpy==1.26.4
   pandas==2.2.3
   streamlit==1.39.0
   requests==2.32.3
   paddlepaddle==2.6.2
   paddleocr==2.8.1
   fuzzywuzzy==0.18.0
   python-Levenshtein==0.25.1
   h5py==3.12.1
   setuptools==75.1.0
   ```

4. **Prepare Dataset**:
   - Place wine.csv in the project root with columns: id, name, spirit_type, image_url, avg_msrp, fair_price, shelf_price, total_score.
   - The app will preprocess images and store data in dataset/ and reference_data.h5 on first run.

5. **Run the App**:
   ```
   streamlit run whiskey.py
   ```
   Access at http://localhost:8501.

### Streamlit Cloud

- Ensure wine.csv, requirements.txt, and packages.txt are in the repository root.
- packages.txt:
  ```
  libgl1-mesa-glx
  libglib2.0-0
  ```

## Usage

### Identify Whisky by Image

1. Navigate to the "Identify by Image" tab.
2. Upload a JPG, PNG, JPEG, or WEBP image of a whisky bottle.
3. View detected label regions in color with green bounding boxes.
4. For each region, see:
   - Bottle details (name, spirit type, prices, score).
   - Confidence score (≥0.7 for high confidence).
   - Extracted text from the label.
   - Warnings for low-confidence matches or text validation issues.
5. Click "Clear Image" to upload a new image.

### Search Database

1. Navigate to the "Search Database" tab.
2. Enter whisky name, select spirit type, choose price type, and set price range.
3. Submit to view matching whiskeys in a table.

## Troubleshooting

### Dependency Errors

- **Local**: Run `pip check` to verify dependency compatibility.
- **Streamlit Cloud**: Check deployment logs for installation errors. Ensure requirements.txt and packages.txt are correct.

### PaddleOCR Issues

- **OCR Fails or Misses Labels**:
  - Lower OCR_CONFIDENCE_THRESHOLD:
    ```python
    OCR_CONFIDENCE_THRESHOLD = 0.7
    ```
  - Disable angle classification for speed (may reduce accuracy for rotated text):
    ```python
    return PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)
    ```
  - Increase max_dim for high-resolution images:
    ```python
    max_dim = 1200
    ```

### Slow Processing

- Reduce MAX_REGIONS:
  ```python
  MAX_REGIONS = 3
  ```
- Lower image resolution:
  ```python
  IMAGE_SIZE = (240, 180)
  ```

### Noisy Text or Skipped Text Validation

- Tighten keyword filtering:
  ```python
  words = [w for w in cleaned_text.split() if len(w) > 3]
  ```
- Add BAXUS-specific keywords:
  ```python
  keywords.update(['distillery', 'aged', 'cask', 'proof'])
  ```
- Lower MIN_TEXT_SIMILARITY:
  ```python
  MIN_TEXT_SIMILARITY = 70
  ```

### Color Display Issues

- Ensure channels="BGR" in st.image calls.
- Verify input images are in color (not grayscale).

### Deployment Errors

- Confirm wine.csv is in the repository.
- Check file paths in whiskey.py (e.g., dataset/, reference_data.h5).

## Future Improvements

### Performance

- Cache OCR and feature matching results for repeated images.
- Explore GPU-enabled PaddleOCR if hardware is available.
- Optimize BFMatcher by pre-filtering reference descriptors using text-based similarity.

### Accuracy

- Train a classifier to filter non-whisky images (e.g., backgrounds, non-label regions).
- Use NLP (e.g., spaCy) for advanced text matching and entity recognition.
- Fine-tune PaddleOCR for whisky-specific text patterns.

### Features

- Add image preprocessing options (e.g., rotation, cropping, brightness adjustment).
- Support batch image uploads for bulk pricing data collection.
- Export search results as CSV for pricing data analysis.
- Display matched bottle images alongside results for visual confirmation.

### Fallback OCR

- Implement Google Cloud Vision API as a fallback for PaddleOCR failures.
- Avoid Tesseract (poor performance) and EasyOCR (PyTorch dependency).

## Demo

The deployed Streamlit app demonstrates identifying whisky bottles from the BAXUS 500-bottle dataset. Upload images to see real-time label detection, text extraction, and bottle matching in color. Access the app at https://whiskey.streamlit.app.

## Demo Video

See Whisky Goggles in action:

[![Whisky Goggles Demo](https://img.youtube.com/vi/_wqAXRh891g/0.jpg)](https://youtu.be/_wqAXRh891g)
