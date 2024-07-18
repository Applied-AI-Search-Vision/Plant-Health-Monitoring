from transformers import pipeline
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

# Load the Depth Anything model
depth_pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# Global variables to store the ROI coordinates
roi = [0, 0, 0, 0]
drawing = False


def draw_rectangle(event, x, y, flags, param):
    global roi, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi[2], roi[3] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi[2], roi[3] = x, y


def select_roi(frame):
    global roi, drawing
    roi = [0, 0, 0, 0]
    drawing = False
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", draw_rectangle)

    while True:
        temp_frame = frame.copy()
        if drawing:
            cv2.rectangle(temp_frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp_frame)
        if cv2.waitKey(1) & 0xFF == 13:  
            break

    cv2.destroyAllWindows()
    x1, y1, x2, y2 = roi
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def load_fastsam_model():
    model_path = "FastSAM-x.pt"  
    model = FastSAM(model_path)  
    return model

model = load_fastsam_model()

# Function to extract segmentation mask using FastSAM for a frame within the ROI
def extract_fastsam_mask(frame, model, roi):
    x1, y1, x2, y2 = roi
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi_frame = frame_rgb[y1:y2, x1:x2]  # Crop to the ROI
    everything_results = model(roi_frame, device="cuda", retina_masks=True, imgsz=1024, conf=0.3, iou=0.85)
    prompt_process = FastSAMPrompt(roi_frame, everything_results, device="cuda")

   
    annotations = prompt_process.everything_prompt()

    if isinstance(annotations, list) and len(annotations) > 0:
        masks = annotations[0].masks  
        if masks is not None and len(masks.data) > 0:
            mask = masks.data[0].cpu().numpy()  
            mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)  
            mask_full[y1:y2, x1:x2] = mask  
            return mask_full
        else:
            print("No valid masks found in annotations.")
            return np.zeros(frame.shape[:2], dtype=np.uint8)
    else:
        print("No valid annotations found.")
        return np.zeros(frame.shape[:2], dtype=np.uint8)

# Function to process the video and extract segmentation masks and depth maps for each frame
def process_video(video_path, model, roi, depth_pipe):
    cap = cv2.VideoCapture(video_path)
    mask_list = []
    depth_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished reading the video or error occurred.")
            break
        mask = extract_fastsam_mask(frame, model, roi)
        mask_list.append(mask)

        # Extract depth map using Depth Anything
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        depth_map = depth_pipe(pil_image)["depth"]
        depth_list.append(depth_map)

        # Draw ROI rectangle on the frame
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert the mask to uint8 before resizing
        mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_rgb = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        combined_frame = cv2.addWeighted(frame, 0.5, mask_rgb, 0.5, 0)
        cv2.imshow('Segmented Frame', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if mask_list and depth_list:
        mask_array = np.array(mask_list)
        depth_array = np.array(depth_list)
        return mask_array, depth_array
    else:
        print('No masks or depth maps were extracted from the video.')
        return None, None

# Function to calculate plant height based on segmentation masks and depth maps
def calculate_plant_height(mask_array, depth_array):
    heights = []
    for mask, depth in zip(mask_array, depth_array):
        plant_pixels = np.where(mask == 1) 
        if len(plant_pixels[0]) > 0:
            plant_depths = depth[plant_pixels]
            top_depth = np.min(plant_depths)
            bottom_depth = np.max(plant_depths)
            height = bottom_depth - top_depth
            heights.append(height)
        else:
            heights.append(0)  # No plant detected in this frame
    # Apply smoothing to reduce noise
    heights = np.array(heights)
    heights_smoothed = np.convolve(heights, np.ones(5)/5, mode='same')
    return heights_smoothed.tolist()

# Main execution flow
video_path = 'C:\\Users\\Stell\\Desktop\\DVA 309\\Test_3.mp4'


cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    ROI = select_roi(frame)
    print(f'Selected ROI: {ROI}')

    mask_array, depth_array = process_video(video_path, model, ROI, depth_pipe)

    if mask_array is not None and depth_array is not None:
        heights = calculate_plant_height(mask_array, depth_array)
        
        # Visualize the change in plant height over time
        plt.figure(figsize=(10, 6))
        plt.plot(heights, marker='o', linestyle='-', color='b')
        plt.title('Change in Plant Height Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Plant Height (depth units)')
        plt.grid(True)
        plt.show()
    else:
        print("No valid data to plot.")
else:
    print("Failed to read the video.")