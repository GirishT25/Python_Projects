import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Streamlit UI Configuration (Move this to the top)
st.set_page_config(page_title="Traffic Car Counter", layout="centered")

# Load YOLOv5 model
@st.cache_data
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the model
model = load_model()

# Define interested vehicle classes
INTERESTED_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

# Traffic suggestion based on vehicle count
def get_suggestion(count):
    if count < 5:
        return "Traffic is light. No change needed in signal timing."
    elif count < 15:
        return "Moderate traffic. Consider extending green light duration by 10 seconds."
    elif count < 30:
        return "Heavy traffic. Suggest dynamic signal timing and alternate route alerts."
    else:
        return "Severe congestion. Trigger emergency management protocols and reroute traffic."

# Traffic Signal Control (Simulated)
def manage_traffic(count):
    if count < 5:
        return "Green light duration: 30 seconds. Traffic flow is smooth."
    elif count < 15:
        return "Green light duration: 40 seconds. Moderate traffic, but manageable."
    elif count < 30:
        return "Green light duration: 60 seconds. Traffic management in progress."
    else:
        return "Green light duration: 120 seconds. Triggering emergency rerouting protocols."

# Detect and annotate vehicles in the image
def detect_and_annotate(image):
    image_np = np.array(image)  # Convert PIL image to numpy array
    results = model(image_np)  # Get YOLOv5 model results

    df = results.pandas().xyxy[0]  # Extract results into a pandas dataframe
    vehicle_df = df[df['name'].isin(INTERESTED_CLASSES)]  # Filter only interested vehicle classes
    count = len(vehicle_df)  # Count the number of vehicles

    # Annotate image with detected vehicles
    for _, row in vehicle_df.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = row['name']
        # Draw rectangle and label for each detected vehicle
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image_np, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Convert the image back to PIL for Streamlit display
    image_pil = Image.fromarray(image_np)
    return image_pil, count

# Streamlit UI
st.title("🚦 Traffic Image Analyzer")
st.write("Upload a traffic image to detect vehicles and get traffic management suggestions.")

# File uploader for traffic image
uploaded_file = st.file_uploader("📤 Upload Traffic Image", type=["jpg", "jpeg", "png"])

# If an image is uploaded, process it
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Open image as RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing traffic..."):
        # Detect vehicles and annotate the image
        annotated_img, car_count = detect_and_annotate(image)

        # Display the annotated image
        st.image(annotated_img, caption=f"Detected Vehicles: {car_count}", use_column_width=True)

        # Display the results
        st.success(f"✅ Total Vehicles Detected: {car_count}")
        st.info(f"📊 Traffic Suggestion: {get_suggestion(car_count)}")
        st.write(f"🚦 Traffic Signal Management: {manage_traffic(car_count)}")

else:
    st.write("Please upload a traffic image to begin analysis.")
