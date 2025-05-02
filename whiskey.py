import cv2
import numpy as np
import pandas as pd
import streamlit as st
import os
import requests
from paddleocr import PaddleOCR
import re
from fuzzywuzzy import fuzz
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import h5py

# Constants
DATASET_PATH = "dataset"
HDF5_PATH = "reference_data.h5"
CSV_FILE = "wine.csv"
THRESHOLD = 30
MIN_KEYPOINTS = 50
MIN_TEXT_SIMILARITY = 80
IMAGE_SIZE = (320, 240)
MAX_REGIONS = 5
CONFIDENCE_THRESHOLD = 0.7
OCR_CONFIDENCE_THRESHOLD = 0.8

# Create dataset directory
os.makedirs(DATASET_PATH, exist_ok=True)

# Load and clean CSV
df = pd.read_csv(CSV_FILE)
df.set_index('id', inplace=True)
df['spirit_type'] = df['spirit_type'].fillna("Unknown").astype(str)
for col in ['avg_msrp', 'fair_price', 'shelf_price']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)



# Extract keywords from wine.csv
def extract_keywords(df):
    df = pd.read_csv(CSV_FILE)
    keywords = set()
  
    for name in df['name'].dropna():
        words = re.findall(r'\b\w+\b', name.lower())
        words = [w for w in words if len(w) >= 3 and not w.isdigit()]
        keywords.update(words)
   # Extract from spirit_type column
    for spirit_type in df['spirit_type'].dropna():
        words = re.findall(r'\b\w+\b', spirit_type.lower())
        words = [w for w in words if len(w) >= 3 and not w.isdigit()]
        keywords.update(words)
    # Add exhaustive whisky-related keywords
    keywords.update([
        'whisky', 'whiskey', 'single', 'malt', 'bourbon', 'scotch', 'reserve', 'distillery', 'aged', 'cask', 'proof',
        'blanton', 'eagle', 'taylor', 'buffalo', 'weller', 'stagg', 'mckenna', 'heaven', 'hill', 'elijah', 'craig',
        'knob', 'creek', 'sazerac', 'wild', 'turkey', 'woodford', 'russell', 'forester', 'rock', 'farms', 'smoke',
        'wagon', 'caribou', 'crossing', 'evan', 'williams', 'four', 'roses', 'angel', 'envy', 'willett', 'michter',
        'booker', 'pappy', 'van', 'winkle', 'traveller', 'george', 'jack', 'daniel', 'blade', 'bow', 'maker', 'mark',
        'high', 'west', 'william', 'larue', 'widow', 'jane', 'penelope', 'baker', 'larceny', 'benchmark', 'yellowstone',
        'ezra', 'brooks', 'double', 'eagle', 'coy', 'basil', 'hayden', 'redwood', 'empire', 'pipe', 'dream', 'lost',
        'monarch', 'hancock', 'president', 'thomas', 'handy', 'peerless', 'cream', 'joseph', 'magnus', 'calumet',
        'dickel', 'little', 'book', 'john', 'bowman', 'noah', 'mill', 'kentucky', 'owl', 'tub', 'new', 'riff', 'uncle',
        'nearest', 'blood', 'oath', 'cooper', 'craft', 'remus', 'repeal', 'belle', 'meade', 'monkey', 'shoulder',
        'james', 'pepper', 'rabbit', 'hole', 'castle', 'key', 'horse', 'soldier', 'blue', 'note', 'run', 'green', 'river',
        'barrell', 'seagrass', 'whistle', 'pig', 'pikesville', 'jefferson', 'lagavulin', 'mellow', 'corn', 'grizzly',
        'beast', 'yamazaki', 'hibiki', 'bib', 'tucker', 'sagamore', 'kirkland', 'evans', 'redbreast', 'crown', 'royal',
        'jameson', 'black', 'spot', 'glenfiddich', 'nikka', 'tito', 'vodka', 'lux', 'row', 'johnnie', 'walker',
        'wilderness', 'trail', 'legent', 'rio', 'frey', 'ranch', 'fortaleza', 'tequila', 'alberta', 'premium', 'chicken',
        'cock', 'bernheim', 'eric', 'church', 'thirteenth', 'colony', 'noble', 'oak', 'woodinville', 'frank', 'august',
        'skrewball', 'peanut', 'butter', 'macallan', 'heaven', 'door', 'garrison', 'brothers', 'balmorhea', 'clase',
        'azul', 'reposado', 'don', 'julio', 'hendrick', 'gin', 'king', 'balvenie', 'glenlivet', 'laphroaig', 'blackened',
        'brother', 'bond', 'rittenhouse', 'naber', 'orphan', 'fabel', 'folly', 'original', 'barrel', 'rare', 'year',
        'small', 'batch', 'antique', 'special', 'full', 'toasted', 'bond', 'bonded', 'straight', 'rye', 'breed', 'oaked',
        'prohibition', 'fine', 'wheated', 'select', 'midwinter', 'night', 'dram', 'act', 'scene', 'sour', 'mash',
        'architect', 'strength', 'sinatra', 'black', 'family', 'statesman', 'decade', 'release', 'crossing', 'unfiltered',
        'uncut', 'maple', 'smoked', 'marriage', 'campfire', 'mighty', 'storyteller', 'apprentice', 'piggyback', 'longbranch',
        'peach', 'salt', 'caramel', 'triple', 'distilled', 'sherry', 'experimental', 'vintage', 'invitation', 'confiscated',
        'anniversary', 'heritage', 'collection', 'soft', 'red', 'wheat', 'wood', 'finish', 'heart', 'kentucky', 'spirit',
        'juke', 'joint', 'private', 'selection', 'emerald', 'giant', 'screaming', 'titan', 'stouted', 'homestead', 'amaranth',
        'blended', 'subtle', 'smoke', 'dovetail', 'very', 'landmark', 'ocean', 'sea', 'voyage', 'french', 'oaked', 'musician',
        'cellar', 'blue', 'label', 'takumi', 'port', 'valencia', 'rosé', 'fusion', 'discovery', 'derby', 'gentleman', 'grain',
        'glass', 'bourye', 'twice', 'barreled', 'oloroso', 'malted', 'path', 'taken', 'rack', 'cigar', 'blend', 'harmony',
        'dark', 'retrospect', 'amburana', 'coffee', 'jts', 'brown', 'declaration', 'halloween', 'coffey', 'crossroads',
        'distiller', 'vault', 'malt', 'experimental', 'barrelhouse', 'cut', 'regal', 'apple', 'infinite', 'project'
    ])
    return keywords

KEYWORDS = extract_keywords(df['name'])

# Add Google Font and custom CSS
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""

    <style>
    stApp {
        background-color: #0000ff; 
    }
    .main {
        background-color: #0000ff;
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
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: scale(1.02);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        background-color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stProgress .st-bo {
        background-color: #8b4513;
    }
    @media (max-width: 600px) {
        .stColumn {
            width: 100% !important;
            margin-bottom: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize PaddleOCR with improved settings
@st.cache_resource
def init_paddle_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_db_score_mode='slow', det_db_box_type='quad')

ocr = init_paddle_ocr()

# Download image
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

# Download image task for ProcessPoolExecutor
def download_image_task(idx, url, path):
    if not os.path.exists(path):
        download_image(url, path)

# Preprocess reference images
def preprocess_references():
    if os.path.exists(HDF5_PATH):
        return
    
    st.info("Preprocessing reference images. This may take a few minutes...")
    
    # Parallel image downloads
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download_image_task, idx, row['image_url'], os.path.join(DATASET_PATH, f"{idx}.jpg"))
            for idx, row in df.iterrows()
        ]
        for future in futures:
            future.result()
    
    # Process and save to HDF5
    with h5py.File(HDF5_PATH, 'w') as f:
        for idx, row in df.iterrows():
            image_path = os.path.join(DATASET_PATH, f"{idx}.jpg")
            if os.path.exists(image_path):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                    img = normalize_image(img)
                    orb = cv2.ORB_create(nfeatures=500)
                    kp, des = orb.detectAndCompute(img, None)
                    if des is not None and len(kp) > 0:
                        f.create_dataset(f"descriptors/{idx}", data=des)
                        kp_data = np.array([
                            (k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
                            for k in kp
                        ], dtype=[
                            ('x', np.float32), ('y', np.float32), ('size', np.float32),
                            ('angle', np.float32), ('response', np.float32),
                            ('octave', np.int32), ('class_id', np.int32)
                        ])
                        f.create_dataset(f"keypoints/{idx}", data=kp_data)
    
    st.success("Preprocessing complete!")

# Load reference data
@st.cache_resource
def load_reference_data():
    ref_descriptors = {}
    ref_keypoints = {}
    with h5py.File(HDF5_PATH, 'r') as f:
        for idx in df.index.astype(str):
            if f"descriptors/{idx}" in f:
                ref_descriptors[int(idx)] = f[f"descriptors/{idx}"][:]
                kp_data = f[f"keypoints/{idx}"][:]
                ref_keypoints[int(idx)] = [
                    cv2.KeyPoint(
                        x=float(k['x']), y=float(k['y']), size=float(k['size']),
                        angle=float(k['angle']), response=float(k['response']),
                        octave=int(k['octave']), class_id=int(k['class_id'])
                    ) for k in kp_data
                ]
    return ref_descriptors, ref_keypoints

# Normalize image (preserve color option)
def normalize_image(image, preserve_color=False):
    if len(image.shape) == 3 and not preserve_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not preserve_color:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
    return image

# Run PaddleOCR with improved label detection
def run_ocr(image):
    try:
        # Keep original color image
        original_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Downsample image for OCR
        max_dim = 1000  # Increased for better text detection
        h, w = image.shape[:2]
        scale = min(max_dim / h, max_dim / w)
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            original_image = cv2.resize(original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        # Convert to RGB for OCR
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Run PaddleOCR
        results = ocr.ocr(img_rgb, cls=True)
        if not results or not results[0]:
            return [], [], [""], original_image
        
        regions, bboxes, texts = [], [], []
        min_area = 0.005 * image.shape[0] * image.shape[1]  # Relaxed for smaller labels
        
        for line in results[0]:
            box, (text, confidence) = line[0], line[1]
            if confidence < OCR_CONFIDENCE_THRESHOLD or len(text.strip()) < 3:
                continue
            
            x_min = int(min(p[0] for p in box))
            y_min = int(min(p[1] for p in box))
            x_max = int(max(p[0] for p in box))
            y_max = int(max(p[1] for p in box))
            w, h = x_max - x_min, y_max - y_min
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            
            if area > min_area and 0.3 < aspect_ratio < 3.0 and w > 30 and h > 20:  # Relaxed filters
                sub_image = original_image[y_min:y_max, x_min:x_max]  # Use color image
                regions.append(sub_image)
                bboxes.append((x_min, y_min, x_max, y_max))
                texts.append(text)
        
        img_with_boxes = original_image.copy()
        for (x_min, y_min, x_max, y_max) in bboxes:
            cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        regions = sorted(regions, key=lambda r: r.shape[0] * r.shape[1], reverse=True)[:MAX_REGIONS]
        return regions, bboxes, texts, img_with_boxes
    except Exception as e:
        #st.warning(f"OCR failed: {e}. Processing entire image.")
        return [original_image], [], [""], original_image

# Compute query descriptors
def compute_query_descriptors(query_img):
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(query_img, None)
    return kp, des

# Clean and match text with relaxed filtering
def clean_and_match_text(extracted_text, whiskey_name):
    if not extracted_text:
        return None
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', extracted_text.lower())
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    words = [w for w in cleaned_text.split() if len(w) > 2]  # Relaxed length filter
    cleaned_text = ' '.join(words)
    if not cleaned_text:
        return None
    
    whiskey_name_clean = re.sub(r'[^a-zA-Z\s]', '', whiskey_name.lower()).strip()
    whiskey_words = [w for w in whiskey_name_clean.split() if len(w) > 2]
    
    text_keywords = set(words)
    whiskey_keywords = set(whiskey_words)
    common_keywords = text_keywords.intersection(whiskey_keywords).intersection(KEYWORDS)
    
    if len(common_keywords) >= 2:
        return True
    elif len(common_keywords) >= 1:
        similarity = fuzz.partial_ratio(cleaned_text, whiskey_name_clean)
        return similarity >= MIN_TEXT_SIMILARITY
    elif cleaned_text and whiskey_name_clean:
        similarity = fuzz.partial_ratio(cleaned_text, whiskey_name_clean)
        return similarity >= MIN_TEXT_SIMILARITY + 10  # Stricter threshold for no common keywords
    return False

# Find best match
def find_best_match(query_kp, query_des, ref_descriptors, ref_keypoints):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    best_match_id = None
    max_good_matches = 0
    best_homography = None

    candidates = []
    for idx, ref_des in ref_descriptors.items():
        if ref_des is not None and query_des is not None:
            matches = bf.knnMatch(query_des, ref_des, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            if len(good_matches) >= 8:
                candidates.append((idx, good_matches))
    
    candidates = sorted(candidates, key=lambda x: len(x[1]), reverse=True)[:5]
    
    for idx, good_matches in candidates:
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

# Identify whiskey
def identify_whiskey(sub_img, ref_descriptors, ref_keypoints, extracted_text=""):
    # Convert color image to grayscale for feature detection
    sub_img_gray = normalize_image(sub_img, preserve_color=False)
    sub_img_resized = cv2.resize(sub_img_gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    query_kp, query_des = compute_query_descriptors(sub_img_resized)

    if query_kp is None or len(query_kp) < MIN_KEYPOINTS:
        return {"status": "insufficient_keypoints", "sub_img": sub_img}

    if query_des is None:
        return {"status": "no_features", "sub_img": sub_img}

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
            "sub_img": sub_img  # Return color sub_img
        }
    return {
        "status": "no_match",
        "num_matches": num_matches,
        "extracted_text": extracted_text,
        "sub_img": sub_img  # Return color sub_img
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
                filtered_df[['name', 'spirit_type', 'avg_msrp', 'size', 'ranking', 'total_score']],
                use_container_width=True
            )

# Streamlit application
def main():
    st.title("Whiskey Googles")
    st.markdown("<h2 style='text-align: center; color: #ffff00;'>Identify bottles or search the database</h2>", unsafe_allow_html=True)

    if not os.path.exists(HDF5_PATH):
        preprocess_references()

    ref_descriptors, ref_keypoints = load_reference_data()

    tab1, tab2 = st.tabs(["📷 Identify by Image", "🔍 Search Database"])

    with tab1:
        st.markdown("Upload an image containing one or more whiskey bottle labels to identify them. For best results, ensure each label is clearly visible and not overlapping.", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "webp"], key="image_uploader")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Load in color
            query_img = normalize_image(query_img, preserve_color=True)  # Preserve color
            
            with st.spinner("Processing image..."):
                regions, bboxes, extracted_texts, img_with_boxes = run_ocr(query_img)
                st.image(img_with_boxes, channels="BGR", caption="Detected Label Regions", use_container_width=True)
                if not regions:
                    st.info("No distinct label regions detected. Processing the entire image.")
                    regions = [query_img]
                    extracted_texts = [""]
                else:
                    st.info(f"Detected {len(regions)} potential whiskey label regions.")
                
                results = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(identify_whiskey, sub_img, ref_descriptors, ref_keypoints, text)
                        for sub_img, text in zip(regions, extracted_texts + [""] * (len(regions) - len(extracted_texts)))
                    ]
                    for future in futures:
                        results.append(future.result())
                
                results = sorted(results, key=lambda x: x.get('confidence', 0), reverse=True)
                
                for i, result in enumerate(results):
                    if result["status"] in ["success", "low_confidence"]:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(result["sub_img"], channels="BGR", caption=f"Detected Region {i+1}", use_container_width=True)
                        with col2:
                            whisky_info = result["whisky_info"]
                            st.markdown(f"<div class='result-card'><h3 style='color: black;'>{whisky_info['name']}</h3>", unsafe_allow_html=True)
                            st.write(f"**Spirit Type:** {whisky_info['spirit_type']}")
                            st.write(f"**Average Price:** ${whisky_info['avg_msrp']:.2f}")
                            st.write(f"**Size:** {whisky_info['size']}ml")
                            st.write(f"**Rank:** {whisky_info['ranking']}")
                            st.write(f"**Total Score:** {whisky_info['total_score']}")
                            st.write(f"**Confidence Score:** {result['confidence']:.2f}")
                            # st.write(f"**Extracted Text:** {result['extracted_text']}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            if result["status"] == "low_confidence":
                                st.warning(f"Low confidence match for {whisky_info['name']}. Verify the result.")
                            if result["text_match"] is None:
                                pass
                               # st.warning("Text validation skipped due to OCR issues. Match based on image features only.")
                    else:
                        st.write(f"Region {i+1}: No match found.")
                        if result["extracted_text"]:
                            pass
                            # st.write(f"**Extracted Text:** {result['extracted_text']}")
        
        if st.button("Clear Image", key="clear_button"):
            st.session_state.pop("image_uploader", None)
            st.rerun()

    with tab2:
        search_whiskeys()

if __name__ == "__main__":
    main()
