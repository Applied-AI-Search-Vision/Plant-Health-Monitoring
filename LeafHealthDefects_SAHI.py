import torch
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    visualize_results,
)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the YOLOv9 model for instance segmentation
yolo_segmentation_model = YOLO('C:\\Users\\Stell\\Desktop\\DVA 309\\runs\\segment\\train5\\weights\\best.pt')
yolo_segmentation_model.conf = 0.8  

def apply_masks_to_frame(frame, masks):
    """
    Applies the segmentation masks to the frame to isolate the foreground (leaves).
    Pixels not covered by the masks are set to black, effectively removing the background.
    """
    frame_masked = frame.copy()
    for mask in masks:
        mask = mask.cpu().numpy().astype(np.uint8)  # Ensure mask is a numpy array of type uint8
        frame_masked = cv2.bitwise_and(frame_masked, frame_masked, mask=mask)
    return frame_masked

def process_video(video_path, yolo_segmentation_model):
    processed_frames = []  # List to store processed frames for later use or visualization

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more video frames or failed to capture video frame")
            break

        # Process frame for detection using patched_yolo_infer
        element_crops = MakeCropsDetectThem(
            image=frame,
            model_path="C:\\Users\\Stell\\Desktop\\DVA 309\\runs\\segment\\train5\\weights\\best.pt",
            segment=True,
            show_crops=False,
            shape_x=600,
            shape_y=500,
            overlap_x=50,
            overlap_y=50,
            conf=0.6,
            iou=0.7,
            classes_list=[0, 1],
            resize_initial_size=True,
        )
        result = CombineDetections(element_crops, nms_threshold=0.05, match_metric='IOS')

        # YOLO-Patch-Based-Inference Display
        processed_frame = visualize_results(
            img=result.image,
            confidences=result.filtered_confidences,
            boxes=result.filtered_boxes,
            masks=result.filtered_masks,  
            classes_ids=result.filtered_classes_id,
            classes_names=result.filtered_classes_names,
            segment=True,
            thickness=8,
            font_scale=1.1,
            fill_mask=False, 
            show_boxes=False,
            delta_colors=3,
            show_class=True,
            axis_off=True,
            return_image_array=True
        )

        for box, conf, cls_name in zip(result.filtered_boxes, result.filtered_confidences, result.filtered_classes_names):
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(processed_frame, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Add the masked frame to the list of processed frames
        processed_frames.append(processed_frame)

        # Display the frame using OpenCV
        cv2.imshow("Detection Output", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close all OpenCV windows

    return processed_frames

def main():
    video_path = 'C:\\Users\\Stell\\Desktop\\DVA 309\\GX010009.MP4'
    processed_frames = process_video(video_path, yolo_segmentation_model)

    
    for i, frame in enumerate(processed_frames):
        cv2.imwrite(f'output/frame_{i}.png', frame)
        cv2.imshow(f'Processed Frame {i}', frame)
        cv2.waitKey(1) 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()