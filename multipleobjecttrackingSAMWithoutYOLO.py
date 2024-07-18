import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO, FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import supervision as sv
from ultralytics.trackers.byte_tracker import STrack

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


fastsam_model = load_fastsam_model()


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

# Function to process the video and extract segmentation masks for each frame
def process_video(video_path, fastsam_model, yolo_model, roi):
    cap = cv2.VideoCapture(video_path)
    mask_list = []
    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished reading the video or error occurred.")
            break
        
        # Extract segmentation mask using FastSAM
        mask = extract_fastsam_mask(frame, fastsam_model, roi)
        mask_list.append(mask)

        # Crop the frame to the ROI
        roi_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]

       
        results = yolo_model(roi_frame)

        # Convert the results to supervision Detections format
        detections = sv.Detections.from_ultralytics(results[0])

        # Perform multi-object prediction using Kalman filter for all tracked objects
        STrack.multi_predict(tracker.tracked_tracks)

       
        tracker.update_with_detections(detections)

        # Visualize the tracking results
        for track in tracker.tracked_tracks:
            tlwh = track.tlwh
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(roi_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(roi_frame, str(track.track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw ROI rectangle on the frame
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
        mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_rgb = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        combined_frame = cv2.addWeighted(frame, 0.5, mask_rgb, 0.5, 0)
        cv2.imshow('Segmented Frame', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if mask_list:
        mask_array = np.array(mask_list)
        print(f'Shape of mask_array: {mask_array.shape}')  # Debugging line
        return mask_array
    else:
        print('No masks were extracted from the video.')
        return None  


def calculate_plant_height(mask_array):
    heights = []
    for mask in mask_array:
        plant_pixels = np.where(mask == 1) 
        if len(plant_pixels[0]) > 0:
            top = min(plant_pixels[0])
            bottom = max(plant_pixels[0])
            height = bottom - top
            heights.append(height)
        else:
            heights.append(0)  
   
    heights = np.array(heights)
    heights_smoothed = np.convolve(heights, np.ones(5)/5, mode='same')
    return heights_smoothed.tolist()

video_path = 'C:\\Users\\Stell\\Desktop\\DVA 309\\GX010009.MP4'


yolo_model = YOLO('bestleafseg.pt')


cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    ROI = select_roi(frame)
    print(f'Selected ROI: {ROI}')

    mask_array = process_video(video_path, fastsam_model, yolo_model, ROI)

    if mask_array is not None:
        heights = calculate_plant_height(mask_array)
        
        # Visualize the change in plant height over time
        plt.figure(figsize=(10, 6))
        plt.plot(heights, marker='o', linestyle='-', color='b')
        plt.title('Change in Plant Height Over Time')
        plt.xlabel('Frame Index')
        plt.ylabel('Plant Height (pixels)')
        plt.grid(True)
        plt.show()
    else:
        print("No valid data to plot.")
else:
    print("Failed to read the video.")