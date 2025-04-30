# Whisky Goggles

**Whisky Goggles** is a Streamlit-based web application developed for BAXUS to scan whisky bottle labels from images and match them to a database of 500 whisky bottles. Using computer vision (OpenCV) and OCR (PaddleOCR), it detects and recognizes labels, extracts text, and identifies bottles with high accuracy, even under varying lighting, angles, and partial labels. Users can also search the database by name, spirit type, and price range to record pricing information at liquor stores.

---

## Overview

The goal of Whisky Goggles is to develop a computer vision system that enables users to:

- Scan whisky bottle labels from photos.
- Match identified labels to BAXUS’s dataset of 500 bottles.
- Provide structured output with confidence scores for matches.
- Facilitate pricing data collection in liquor stores.

---


## Features

- **Label Detection & Recognition**:
  - Processes images of whisky bottle labels using PaddleOCR to detect text regions.
  - Extracts visual features using ORB (Oriented FAST and Rotated BRIEF).
  - Matches labels to the BAXUS dataset with keyword-based text filtering and fuzzy matching.
- **Bottle Identification**:
  - Achieves high accuracy for the 500-bottle dataset.
  - Provides confidence scores combining feature matching (60%) and text matching (40%).
  - Displays low-confidence matches with warnings for user verification.
  - Handles variations in lighting, angles, and partial labels.
- **Database Search**:
  - Search whiskeys by name, spirit type, and price range (MSRP, fair price, shelf price).
  - Presents results in a clean, tabular format.
- **User Interface**:
  - Built with Streamlit for an intuitive, responsive UI.
  - Supports image uploads, a clear button, and styled output with custom CSS and Google Fonts.
- **Output Format**:
  - Structured data including bottle name, spirit type, prices, score, confidence score, and extracted text.
- **Performance**:
  - Uses parallel processing (`ThreadPoolExecutor`) for multiple label regions.
  - Sorts results by confidence for better usability.
- **Noise Reduction**:
  - Filters OCR output to remove non-alphabetic characters and short words (<4 characters).
  - Requires high-confidence OCR results (≥0.8) to minimize errors.

---

## Technical Requirements

### Core Functionality

- **Label Detection & Recognition**:
  - Processes images to detect whisky bottle labels.
  - Extracts key visual features (ORB descriptors) and text (PaddleOCR).
  - Matches labels to the BAXUS 500-bottle dataset.
- **Bottle Identification**:
  - Achieves high accuracy with robust handling of lighting, angle, and partial label variations.
  - Outputs structured data with confidence scores.

### Technical Specifications

- **Computer Vision Approach**:
  - **Feature Matching**: ORB for visual feature extraction and matching.
  - **OCR**: PaddleOCR for text detection and extraction.
  - **Text Matching**: Keyword-based scoring (requires ≥2 matching keywords) and fuzzy matching (≥80% similarity).
- **Output Format**:
  - JSON-like structure with bottle details (name, spirit type, prices, score), confidence score, and extracted text.

---

## Implementation Details

### Technologies

| Component            | Technology                     | Purpose                              |
|----------------------|--------------------------------|--------------------------------------|
| Image Processing     | `opencv-python-headless`       | ORB feature detection, preprocessing |
| OCR                  | `paddleocr`                    | Label detection, text extraction     |
| Data Handling        | `numpy`, `pandas`              | CSV processing, data manipulation    |
| Web Framework        | `streamlit`                    | User interface and app deployment    |
| Image Downloading    | `requests`                     | Fetch images from dataset URLs       |
| Text Matching        | `fuzzywuzzy`, `python-Levenshtein` | Fuzzy text matching              |
| Parallel Processing  | `concurrent.futures`           | Process multiple label regions       |
| System Dependencies  | `libgl1-mesa-glx`, `libglib2.0-0` | OpenCV and PaddleOCR support     |

- **Dataset**:
  - `wine.csv`: Contains 500 whiskey records with columns: `id`, `name`, `spirit_type`, `image_url`, `avg_msrp`, `fair_price`, `shelf_price`, `total_score`.
  - Preprocessed images stored in `dataset/`, ORB descriptors in `descriptors/`, and keypoints in `keypoints/`.

### Workflow

1. **Preprocessing**:
   - Downloads images from `image_url` in `wine.csv`.
   - Resizes images to 640x480 and normalizes using CLAHE (Contrast Limited Adaptive Histogram Equalization).
   - Extracts ORB keypoints and descriptors, saved to `descriptors/` and `keypoints/`.
2. **Label Detection**:
   - Uses PaddleOCR to detect text regions in uploaded images.
   - Filters regions by area, aspect ratio (0.5–2.0), and OCR confidence (≥0.8).
   - Draws green bounding boxes around detected labels.
3. **Text Extraction and Matching**:
   - Extracts text using PaddleOCR, combining high-confidence results.
   - Cleans text by removing non-alphabetic characters and words <4 characters.
   - Matches text against whiskey names using:
     - Keyword-based scoring (≥2 common keywords from a predefined set).
     - Fuzzy matching (≥80% similarity) if only one keyword matches.
4. **Bottle Identification**:
   - Computes ORB descriptors for label regions.
   - Matches descriptors against preprocessed dataset using BFMatcher with homography (RANSAC).
   - Combines feature confidence (based on number of matches) and text confidence (1.0 for match, 0.5 for OCR failure, 0.0 for mismatch).
   - Outputs matches with confidence ≥0.7 as "success" and lower as "low_confidence" with warnings.
5. **Search Functionality**:
   - Filters `wine.csv` by name, spirit type, and price range.
   - Displays results in a Streamlit dataframe.
6. **Parallel Processing**:
   - Processes multiple label regions concurrently using `ThreadPoolExecutor`.
   - Sorts results by confidence for user-friendly display.

### Key Parameters

| Parameter                  | Value      | Description                                      |
|----------------------------|------------|--------------------------------------------------|
| `THRESHOLD`                | 30         | Min good feature matches for identification      |
| `MIN_KEYPOINTS`            | 50         | Min keypoints for feature matching               |
| `MIN_TEXT_SIMILARITY`      | 80         | Min fuzzy matching score (%)                     |
| `IMAGE_SIZE`               | (640, 480) | Standardized image resolution                    |
| `MAX_REGIONS`              | 5          | Max label regions to process                     |
| `CONFIDENCE_THRESHOLD`     | 0.7        | Min combined confidence for "success" matches    |
| `OCR_CONFIDENCE_THRESHOLD` | 0.8        | Min PaddleOCR confidence for text                |

---


## Prerequisites

- **Python**: 3.8 or higher.
- **System Dependencies** (local development):
  - Linux: `libgl1-mesa-glx`, `libglib2.0-0`.
- **Streamlit Cloud**: Configured via `packages.txt`.

---

## Setup Instructions

### Local Development

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/OMERHRR/Whiskey-Goggles.git
   cd whiskey-goggles
   ```

2. Install System Dependencies (Linux):
  ```bash
  sudo apt-get update
  sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
  ```
3. Install Python Dependencies:
  ```bash
  pip install -r requirements.txt
  ```
    Contents of requirements.txt:
   ``` text
    
    opencv-python-headless==4.8.1.78
    numpy==1.26.4
    pandas==2.2.2
    streamlit==1.38.0
    requests==2.32.3
    paddlepaddle==2.6.1
    paddleocr==2.8.1
    fuzzywuzzy==0.18.0
    python-Levenshtein==0.25.1
    setuptools==70.0.0
  ```
4. Prepare Dataset:
  Place wine.csv in the project root with columns: id, name, spirit_type, image_url, avg_msrp, fair_price, shelf_price, total_score.
  
  Run the App:
  ```bash
  streamlit run whiskey.py
  ```
  Access at http://localhost:8501.

## Usage
**Identify Whisky by Image:**
- Navigate to the "Identify by Image" tab.

- Upload a JPG, PNG, or JPEG image of a whisky bottle.

- View detected label regions with green bounding boxes.

- For each region, see:
  - Bottle details (name, spirit type, prices, score).

  - Confidence score (≥0.7 for high confidence).

  - Extracted text from the label.

  - Warnings for low-confidence matches or text mismatches.

  - Click "Clear Image" to upload a new image.

**Search Database:**
- Navigate to the "Search Database" tab.

- Enter whisky name, select spirit type, choose price type, and set price range.

- Submit to view matching whiskeys in a table.



## Troubleshooting
Dependency Errors:
- Local: Run pip check to verify dependency compatibility.

- Streamlit Cloud: Check logs for installation errors. Ensure requirements.txt and packages.txt are correct.

PaddleOCR Issues:
- If OCR fails, increase OCR_CONFIDENCE_THRESHOLD in whiskey.py:
```python
OCR_CONFIDENCE_THRESHOLD = 0.9
```
- Disable angle classification for speed:
```python
return PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)
```
Slow Processing:
- Reduce MAX_REGIONS:
```python
MAX_REGIONS = 3
```
Lower image resolution:
```python
IMAGE_SIZE = (320, 240)
```
Noisy Text:
-Tighten keyword filtering:
```python
words = [w for w in cleaned_text.split() if len(w) > 4]
```
- Add BAXUS-specific keywords:
```python
keywords.update(['distillery', 'aged', 'cask'])
```
Deployment Errors:
- Verify wine.csv is in the repository.
- Check file paths in whiskey.py.


## Future Improvements
Performance:
- Cache OCR results for repeated images.
- Explore GPU-enabled PaddleOCR if Streamlit Cloud supports it.

Accuracy:
- Train a classifier to filter non-whisky images.
- Use NLP for advanced text matching.

Features:
- Add image preprocessing options (e.g., rotation, cropping).
- Support batch image uploads for bulk pricing data collection.
- Export search results as CSV for pricing data.

Fallback OCR:
Implement Google Vission for PaddleOCR failures. ( Tesseract is a terrible option, EasyOCR heavily depend on pytorch)

# Demo
The deployed Streamlit app demonstrates identifying whisky bottles from the BAXUS 500-bottle dataset. Upload images to see real-time label detection, text extraction, and bottle matching. Access the app at https://whiskey.streamlit.app.




