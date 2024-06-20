import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from streamlit_drawable_canvas import st_canvas

# Load the Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO labels map
COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Initialize session state
if "stop_webcam" not in st.session_state:
    st.session_state.stop_webcam = False

# Function to perform object detection
def detect_objects(image, model, threshold=0.5, selected_classes=None):
    image_tensor = F.to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
    
    pred_classes = outputs[0]['labels'].numpy()
    pred_scores = outputs[0]['scores'].detach().numpy()
    pred_boxes = outputs[0]['boxes'].detach().numpy()
    
    # Filter out low confidence detections
    valid_indices = np.where(pred_scores >= threshold)[0]
    
    if selected_classes is not None:
        valid_indices = valid_indices[np.isin(pred_classes[valid_indices], selected_classes)]
    
    return pred_boxes[valid_indices], pred_classes[valid_indices], pred_scores[valid_indices]

# Function to visualize detections
def visualize_detections(image, boxes, classes, scores, labels_map):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=2))
        
        if 0 < cls < len(labels_map):
            text = f"{labels_map[cls]}: {score:.2f}"
        else:
            text = f"Unknown: {score:.2f}"
        
        ax.text(xmin, ymin, text, bbox=dict(facecolor='yellow', alpha=0.5))
    st.pyplot(plt)

# Function to get image download link
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    buffered.seek(0)
    b64 = base64.b64encode(buffered.read()).decode()
    href = f'<a href="data:image/jpeg;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Callback to stop webcam
def stop_webcam_callback():
    st.session_state.stop_webcam = True

# Function for webcam object detection
def webcam_object_detection(model, threshold=0.5, selected_classes=None):
    stframe = st.empty()

    if st.button("Start Webcam", key="start_webcam"):
        st.session_state.stop_webcam = False

    if st.button("Stop Webcam", key="stop_webcam", on_click=stop_webcam_callback):
        pass  # The actual stop action is handled by the callback

    cap = cv2.VideoCapture(0)
    
    while not st.session_state.stop_webcam:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        boxes, classes, scores = detect_objects(image, model, threshold, selected_classes)
        
        for box, cls, score in zip(boxes, classes, scores):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            if 0 < cls < len(COCO_LABELS):
                text = f"{COCO_LABELS[cls]}: {score:.2f}"
            else:
                text = f"Unknown: {score:.2f}"
            cv2.putText(frame, text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        stframe.image(frame, channels="BGR")
        
        # Check the stop button status
        if st.session_state.stop_webcam:
            break

    cap.release()
    st.write("Webcam stopped.")

# Streamlit app layout
st.title("Object Detection with Faster R-CNN")

# Sidebar for settings
st.sidebar.title("Settings")
threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5)
selected_classes = st.sidebar.multiselect("Select Classes for Detection", COCO_LABELS[1:], default=COCO_LABELS[1:5])
selected_class_indices = [COCO_LABELS.index(cls) for cls in selected_classes]

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Detecting objects...")
    boxes, classes, scores = detect_objects(image, model, threshold, selected_class_indices)
    visualize_detections(image, boxes, classes, scores, COCO_LABELS)

    if st.button("Download Image with Detections"):
        result_image = Image.fromarray(np.array(image))
        st.markdown(get_image_download_link(result_image, "detected_image.jpg", "Download Image"), unsafe_allow_html=True)
    
    detected_objects = pd.DataFrame({
        "Class": [COCO_LABELS[cls] for cls in classes],
        "Confidence Score": scores,
        "Bounding Box": [box.tolist() for box in boxes]
    })
    st.dataframe(detected_objects)

# Webcam detection
if st.sidebar.checkbox("Enable Webcam Detection"):
    webcam_object_detection(model, threshold, selected_class_indices)

# Canvas for drawing bounding boxes
st.sidebar.write("Draw Bounding Boxes")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=2,
    stroke_color="#FF0000",
    background_color="#FFFFFF",
    height=500,
    width=700,
    drawing_mode="rect",
    key="canvas"
)

if st.sidebar.button("Submit Bounding Boxes"):
    st.sidebar.write("Bounding boxes:", canvas_result.json_data["objects"])

# Documentation
st.sidebar.title("Documentation")
st.sidebar.markdown("""
### How to Use
1. Upload an image or enable webcam for live detection.
2. Adjust the confidence threshold to filter detections.
3. Select specific classes to focus on particular objects.
4. Download the results or export them for further analysis.

### Examples
- Upload an image of a crowded street to detect vehicles and pedestrians.
- Use the webcam to detect objects in real-time around your desk.
""")

