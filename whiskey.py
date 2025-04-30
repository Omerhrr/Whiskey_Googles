import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
import requests
import pytesseract
import re
from fuzzywuzzy import fuzz
from collections import Counter

# Set Tesseract path (update for your system if needed)
# For Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For Linux/Mac, Tesseract is usually in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Uncomment and adjust if needed

# Constants
DATASET_PATH = "dataset"
DESCRIPTORS_PATH = "descriptors"
KEYPOINTS_PATH = "keypoints"
CSV_FILE = "wine.csv"
THRESHOLD = 30  # Stricter matching
MIN_KEYPOINTS = 50  # Increased for robustness
MIN_TEXT_SIMILARITY = 80  # Stricter text matching
IMAGE_SIZE = (640, 480)
MAX_REGIONS = 5  # Limit regions to process
CONFIDENCE_THRESHOLD = 0.7  # Minimum combined confidence

# Create directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(DESCRIPTORS_PATH, exist_ok=True)
os.makedirs(KEYPOINTS_PATH, exist_ok=True)

# Load and clean CSV
df = pd.read_csv(CSV_FILE)
df.set_index('id', inplace=True)
df['spirit_type'] = df['spirit_type'].fillna("Unknown").astype(str)
for col in ['avg_msrp', 'fair_price', 'shelf_price']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# Extract keywords from whisky names
def extract_keywords(names):
    keywords = set()
    for name in names:
        words = re.findall(r'\b\w+\b', name.lower())
        words = [w for w in words if len(w) > 3 and not w.isdigit()]
        keywords.update(words)
    keywords.update(['whisky', 'whiskey', 'single', 'malt', 'bourbon', 'scotch', 'reserve'])
    return keywords

KEYWORDS = extract_keywords(df['name'])

# Add Google Font and custom CSS for styling
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #4a2c0b;
    }
    .stButton>button {
        background-color: #8b4513;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #a0522d;
    }
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div {
        border-radius: 5px;
        border: 1px solid #8b4513;
    }
    .result-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Function to download images
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            st.warning(f"Failed to download {url}")
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")

# Preprocess reference images
def preprocess_references():
    progress_bar = st.progress(0)
    total_images = len(df)
    for i, (idx, row) in enumerate(df.iterrows()):
        image_url = row['image_url']
        image_path = os.path.join(DATASET_PATH, f"{idx}.jpg")
        descriptor_path = os.path.join(DESCRIPTORS_PATH, f"{idx}.npy")
        keypoint_path = os.path.join(KEYPOINTS_PATH, f"{idx}_kp.npy")

        if os.path.exists(descriptor_path) and os.path.exists(keypoint_path):
            continue
        
        if not os.path.exists(image_path):
            download_image(image_url, image_path)
        
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                try:
                    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    img = normalize_image(img)
                    orb = cv2.ORB_create()
                    kp, des = orb.detectAndCompute(img, None)
                    if des is not None and len(kp) > 0:
                        np.save(descriptor_path, des)
                        kp_data = np.array([
                            (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
                            for k in kp
                        ], dtype=[
                            ('x', np.float32), ('y', np.float32), ('size', np.float32),
                            ('angle', np.float32), ('response', np.float32),
                            ('octave', np.int32), ('class_id', np.int32)
                        ])
                        np.save(keypoint_path, kp_data)
                    else:
                        st.warning(f"No descriptors or keypoints found for ID {idx}")
                except Exception as e:
                    st.error(f"Error processing image ID {idx}: {e}")
            else:
                st.warning(f"Failed to load image for ID {idx}")
        
        progress_bar.progress((i + 1) / total_images)

# Load reference data
def load_reference_data():
    ref_descriptors = {}
    ref_keypoints = {}
    for idx in df.index:
        descriptor_path = os.path.join(DESCRIPTORS_PATH, f"{idx}.npy")
        keypoint_path = os.path.join(KEYPOINTS_PATH, f"{idx}_kp.npy")
        if os.path.exists(descriptor_path) and os.path.exists(keypoint_path):
            ref_descriptors[idx] = np.load(descriptor_path)
            kp_data = np.load(keypoint_path)
            ref_keypoints[idx] = [
                cv2.KeyPoint(
                    x=float(k['x']), y=float(k['y']), size=float(k['size']),
                    angle=float(k['angle']), response=float(k['response']),
                    octave=int(k['octave']), class_id=int(k['class_id'])
                ) for k in kp_data
            ]
    return ref_descriptors, ref_keypoints

# Normalize image for consistent lighting
def normalize_image(image):
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image)

# Detect potential label regions using contours and Tesseract
def detect_label_regions(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    bboxes = []
    min_area = 0.01 * image.shape[0] * image.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0 and w > 100 and h > 100:
                sub_image = image[y:y+h, x:x+w]
                text = extract_text(sub_image)
                if text and len(text.split()) > 2:
                    regions.append(sub_image)
                    bboxes.append((x, y, x+w, y+h))
    
    img_with_boxes = image.copy()
    for (x_min, y_min, x_max, y_max) in bboxes:
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    regions = sorted(regions, key=lambda r: r.shape[0] * r.shape[1], reverse=True)[:MAX_REGIONS]
    return regions, img_with_boxes

# Compute query descriptors
def compute_query_descriptors(query_img):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(query_img, None)
    return kp, des

# Preprocess image for OCR
def preprocess_image_for_ocr(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.fastNlMeansDenoising(image, h=15)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return image

# Extract text using Tesseract
def extract_text(image):
    try:
        image = preprocess_image_for_ocr(image)
        text = pytesseract.image_to_string(image, lang='eng', config='--psm 6').strip()
        if not text:
            text = pytesseract.image_to_string(image, lang='eng', config='--psm 3').strip()
        return text
    except Exception as e:
        st.warning(f"OCR failed: {e}. Proceeding with feature-based matching only.")
        return ""

# Clean and match text with keyword focus
def clean_and_match_text(extracted_text, whiskey_name):
    if not extracted_text:
        return None
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', extracted_text.lower())
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    words = [w for w in cleaned_text.split() if len(w) > 3]
    cleaned_text = ' '.join(words)
    if not cleaned_text:
        return None
    
    whiskey_name_clean = re.sub(r'[^a-zA-Z\s]', '', whiskey_name.lower()).strip()
    whiskey_words = [w for w in whiskey_name_clean.split() if len(w) > 3]
    
    text_keywords = set(words)
    whiskey_keywords = set(whiskey_words)
    common_keywords = text_keywords.intersection(whiskey_keywords).intersection(KEYWORDS)
    
    if len(common_keywords) >= 2:
        return True
    elif len(common_keywords) == 1:
        similarity = fuzz.partial_ratio(cleaned_text, whiskey_name_clean)
        return similarity >= MIN_TEXT_SIMILARITY
    return False

# Find best match
def find_best_match(query_kp, query_des, ref_descriptors, ref_keypoints):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match_id = None
    max_good_matches = 0
    best_homography = None

    for idx, ref_des in ref_descriptors.items():
        if ref_des is not None and query_des is not None:
            matches = matcher.match(query_des, ref_des)
            good_matches = [m for m in matches if m.distance < 64]
            
            if len(good_matches) > 8:
                ref_kp = ref_keypoints[idx]
                src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0)
                if M is not None:
                    inliers = np.sum(mask)
                    if inliers > max_good_matches:
                        max_good_matches = inliers
                        best_match_id = idx
                        best_homography = M

    return best_match_id, max_good_matches

# Validate if image likely contains a whiskey bottle
def is_valid_bottle_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    text = pytesseract.image_to_string(gray, lang='eng', config='--psm 6').strip()
    return len(text.split()) > 0

# Identify whiskey from sub-image
def identify_whiskey(sub_img, ref_descriptors, ref_keypoints):
    sub_img = normalize_image(sub_img)
    sub_img_resized = cv2.resize(sub_img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    query_kp, query_des = compute_query_descriptors(sub_img_resized)

    if query_kp is None or len(query_kp) < MIN_KEYPOINTS:
        return {"status": "insufficient_keypoints", "sub_img": sub_img}

    if query_des is None:
        return {"status": "no_features", "sub_img": sub_img}

    extracted_text = extract_text(sub_img_resized)
    best_match_id, num_matches = find_best_match(query_kp, query_des, ref_descriptors, ref_keypoints)

    if best_match_id is not None and num_matches >= THRESHOLD:
        whisky_info = df.loc[best_match_id]
        text_match = clean_and_match_text(extracted_text, whisky_info['name'])
        feature_confidence = min(num_matches / 50, 1.0)
        text_confidence = 1.0 if text_match else 0.5 if text_match is None else 0.0
        combined_confidence = 0.6 * feature_confidence + 0.4 * text_confidence
        status = "success" if combined_confidence >= CONFIDENCE_THRESHOLD else "low_confidence"
        return {
            "status": status,
            "whisky_info": whisky_info,
            "num_matches": num_matches,
            "extracted_text": extracted_text,
            "text_match": text_match,
            "confidence": combined_confidence,
            "sub_img": sub_img
        }
    return {
        "status": "no_match",
        "num_matches": num_matches,
        "extracted_text": extracted_text,
        "sub_img": sub_img
    }

# Search functionality
def search_whiskeys():
    st.subheader("Search Whiskeys")
    with st.form(key="search_form"):
        col1, col2 = st.columns(2)
        with col1:
            name_query = st.text_input("Whiskey Name", placeholder="e.g., Woodford Reserve")
            spirit_types = [t for t in df['spirit_type'].unique() if t and t != "Unknown"]
            spirit_type = st.selectbox("Spirit Type", ["All"] + sorted(spirit_types))
        with col2:
            price_type = st.selectbox("Price Type", ["avg_msrp", "fair_price", "shelf_price"])
            min_price = st.number_input("Min Price", min_value=0.0, value=0.0, step=10.0)
            max_price = st.number_input("Max Price", min_value=0.0, value=1000.0, step=10.0)
        submit_button = st.form_submit_button("Search")

    if submit_button:
        if min_price > max_price:
            st.error("Min Price cannot be greater than Max Price.")
            return
        
        filtered_df = df.copy()
        if name_query:
            filtered_df = filtered_df[filtered_df['name'].str.contains(name_query, case=False, na=False, regex=False)]
        if spirit_type != "All":
            filtered_df = filtered_df[filtered_df['spirit_type'] == spirit_type]
        if min_price > 0 or max_price < 1000:
            filtered_df = filtered_df[
                (filtered_df[price_type] >= min_price) & (filtered_df[price_type] <= max_price)
            ]
        
        if filtered_df.empty:
            st.warning("No whiskeys found matching your criteria.")
        else:
            st.write(f"Found {len(filtered_df)} whiskeys:")
            st.dataframe(
                filtered_df[['name', 'spirit_type', 'avg_msrp', 'fair_price', 'shelf_price', 'total_score']],
                use_container_width=True
            )

# Streamlit application
def main():
    st.title("Whiskey Googles")
    st.markdown("<h2 style='text-align: center; color: #4a2c0b;'>Identify bottles or search the database</h2>", unsafe_allow_html=True)

    if not os.path.exists(DESCRIPTORS_PATH) or len(os.listdir(DESCRIPTORS_PATH)) < 500 or not os.path.exists(KEYPOINTS_PATH) or len(os.listdir(KEYPOINTS_PATH)) < 500:
        st.info("Preprocessing reference images. This may take a few minutes...")
        preprocess_references()
        st.success("Preprocessing complete!")

    ref_descriptors, ref_keypoints = load_reference_data()

    tab1, tab2 = st.tabs(["📷 Identify by Image", "🔍 Search Database"])

    with tab1:
        st.markdown("Upload an image containing one or more whiskey bottle labels to identify them. For best results, ensure each label is clearly visible and not overlapping.", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if not is_valid_bottle_image(query_img):
                st.error("This image does not appear to contain a whiskey bottle label.")
            else:
                with st.spinner("Processing image..."):
                    regions, img_with_boxes = detect_label_regions(query_img)
                    st.image(img_with_boxes, channels="BGR", caption="Detected Label Regions", use_container_width=True)
                    if len(regions) == 0:
                        st.info("No distinct label regions detected. Processing the entire image.")
                        regions = [query_img]
                    else:
                        st.info(f"Detected {len(regions)} potential whiskey label regions.")
                    
                    results = []
                    for sub_img in regions:
                        result = identify_whiskey(sub_img, ref_descriptors, ref_keypoints)
                        results.append(result)
                
                for i, result in enumerate(results):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(result["sub_img"], channels="BGR" if len(result["sub_img"].shape) == 3 else "GRAY", caption=f"Detected Region {i+1}", use_container_width=True)
                    with col2:
                        if result["status"] in ["success", "low_confidence"]:
                            whisky_info = result["whisky_info"]
                            st.markdown(f"<div class='result-card'><h3 style='color: black;'>{whisky_info['name']}</h3>", unsafe_allow_html=True)
                            st.write(f"**Spirit Type:** {whisky_info['spirit_type']}")
                            st.write(f"**Average MSRP:** ${whisky_info['avg_msrp']:.2f}")
                            st.write(f"**Fair Price:** ${whisky_info['fair_price']:.2f}")
                            st.write(f"**Shelf Price:** ${whisky_info['shelf_price']:.2f}")
                            st.write(f"**Total Score:** {whisky_info['total_score']}")
                            st.write(f"**Confidence Score:** {result['confidence']:.2f}")
                            # st.write(f"**Extracted Text:** {result['extracted_text']}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            if result["status"] == "low_confidence":
                                st.warning(f"Low confidence match for {whisky_info['name']}. Verify the result.")
                            if result["text_match"] is None:
                                st.warning("Text validation skipped due to OCR issues. Match based on image features only.")
                            elif not result["text_match"]:
                                pass
                                #st.warning(f"Text does not match (extracted: '{result['extracted_text']}'), but image features suggest {whisky_info['name']}.")
                        elif result["status"] == "insufficient_keypoints":
                            st.write("Insufficient keypoints detected in this region.")
                        elif result["status"] == "no_features":
                            st.write("No features detected in this region.")
                        else:
                            st.write("No match found for this region.")
                            st.write(f"**Extracted Text:** {result['extracted_text']}")

    with tab2:
        search_whiskeys()

if __name__ == "__main__":
    main()