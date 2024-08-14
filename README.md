pip install ifnude
Example
Note: Importing ifnude for the first time will download a 139MB module to "/your/home/dir/.ifnude/", just once.

from ifnude import detect

# use mode="fast" for x3 speed with slightly lower accuracy
print(detect('/path/to/nsfw.png'))
Instead of a path, you can use a variable that contains an image loaded through cv2 (opencv) or PIL (pillow).

Output
[
  {'box': [164, 188, 246, 271], 'score': 0.8253238201141357, 'label': 'EXPOSED_BREAST_F'},
  {'box': [252, 190, 335, 270], 'score': 0.8235630989074707, 'label': 'EXPOSED_BREAST_F'}
]

replace the below function from the installed package from the given path: env\Lib\site-packages\ifnude\detector.py 


import os
import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from tqdm import tqdm
import urllib.request

from .detector_utils import preprocess_image


def dummy(*args, **kwargs):
    pass

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

model_url = "https://huggingface.co/s0md3v/nudity-checker/resolve/main/detector.onnx"
classes_url = "https://huggingface.co/s0md3v/nudity-checker/resolve/main/classes"


home = Path.home()
model_folder = os.path.join(home, f".ifnude/")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_name = os.path.basename(model_url)
model_path = os.path.join(model_folder, model_name)
classes_path = os.path.join(model_folder, "classes")

if not os.path.exists(model_path):
    print("Downloading the detection model to", model_path)
    download(model_url, model_path)

if not os.path.exists(classes_path):
    print("Downloading the classes list to", classes_path)
    download(classes_url, classes_path)

classes = [c.strip() for c in open(classes_path).readlines() if c.strip()]

def detect(img, mode="default", min_prob=None):
    # Load the detection model
    detection_model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Set the preprocessing parameters based on the mode
    if mode == "fast":
        image, scale = preprocess_image(img, min_side=480, max_side=800)
        if min_prob is None:
            min_prob = 0.6  # Lower threshold for fast mode
    else:
        image, scale = preprocess_image(img)
        if min_prob is None:
            min_prob = 0.1 # Default threshold

    # Run the model to get predictions
    outputs = detection_model.run(
        [s_i.name for s_i in detection_model.get_outputs()],
        {detection_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
    )

    # Extract labels, scores, and boxes from outputs
    labels = [op for op in outputs if op.dtype == "int32"][0]
    scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
    boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

    # Adjust boxes according to the image scale
    boxes /= scale
    processed_boxes = []

    # Define nudity-related labels with focus on exposed chests
    nudity_labels = {"NUDE",  "EXPOSED_CHEST", "EXPOSED_GENITALIA_M", "EXPOSED_GENITALIA_F" ,"EXPOSED_BREAST_F", "EXPOSED_BUTTOCKS"}
    #"EXPOSED_BELLY" removed for now
    breast_label = "EXPOSED_CHEST"

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < min_prob:
            continue
        box = box.astype(int).tolist()
        label_name = classes[label]

        # Consider only nudity-related labels and skip others
        if label_name not in nudity_labels:
            continue

        # Additional checks specifically for female breasts
        # if label_name == breast_label:
        #     # Enhanced logic to handle partial exposure and nipples
        #     if not is_valid_partial_breast_detection(box, image):
        #         continue
        
        processed_boxes.append(
            {"box": [int(c) for c in box], "score": float(score), "label": label_name}
        )

    return processed_boxes

def is_valid_partial_breast_detection(box, image):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height

    # Ensure aspect ratio is within a reasonable range
    if not (0.5 < aspect_ratio < 2.0):
        return False

    # Check skin tone continuity and symmetry
    region = image[y1:y2, x1:x2]
    if not (check_skin_tone_continuity(region) and check_symmetry(region)):
        return False

    # Additional checks using edge detection and texture
    if not (check_edge_features(region) and check_texture(region)):
        return False

    # Check for nipple exposure
    if not check_nipple_exposure(region):
        return False

    return True

def check_skin_tone_continuity(region):
    # Use color clustering to verify consistent skin tone
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_region, np.array([0, 30, 60]), np.array([20, 150, 255]))
    ratio_skin = cv2.countNonZero(mask) / (region.size / 3)

    # A threshold for determining skin tone consistency
    return ratio_skin > 0.5

def check_symmetry(region):
    # Check symmetry by comparing left and right halves
    h, w = region.shape[:2]
    left_half = region[:, :w // 2]
    right_half = cv2.flip(region[:, w // 2:], 1)

    # Use Structural Similarity Index (SSIM) to measure similarity
    similarity = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0, 0]
    return similarity > 0.8

def check_edge_features(region):
    # Use edge detection to check for clear boundaries
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (region.size / 3)

    # A threshold for determining edge density
    return edge_density > 0.02

def check_texture(region):
    # Check texture using Local Binary Patterns (LBP)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    # A simple check to see if texture matches typical skin patterns
    return hist[0] > 0.1

def check_nipple_exposure(region):
    # Implement a specific logic to check for exposed nipples
    # This can include color detection, texture features, and shape analysis
    nipple_color_lower = np.array([5, 30, 30])  # Adjust based on typical nipple color
    nipple_color_upper = np.array([25, 150, 150])

    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    nipple_mask = cv2.inRange(hsv_region, nipple_color_lower, nipple_color_upper)
    nipple_area_ratio = cv2.countNonZero(nipple_mask) / (region.size / 3)

    # A threshold for determining nipple visibility
    return nipple_area_ratio > 0.1

def local_binary_pattern(image, P, R, method):
    # Function to compute Local Binary Pattern
    # Add your LBP computation logic here
    pass
# Example implementation of an advanced model setup
def advanced_model_setup():
    # Use a pre-trained advanced model (e.g., YOLOv5)
    # Fine-tune it with a specific dataset for partial exposure
    pass


def censor(img_path, out_path=None, visualize=False, parts_to_blur=[]):
    if not out_path and not visualize:
        print(
            "No out_path passed and visualize is set to false. There is no point in running this function then."
        )
        return

    image = cv2.imread(img_path)
    boxes = detect(img_path)

    if parts_to_blur:
        boxes = [i["box"] for i in boxes if i["label"] in parts_to_blur]
    else:
        boxes = [i["box"] for i in boxes]

    for box in boxes:
        image = cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), cv2.FILLED
        )

    return image
