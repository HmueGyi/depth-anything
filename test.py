import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# 1. Model များ Load လုပ်ခြင်း
# YOLO11s model
yolo_model = YOLO("yolo11s.pt") 

# Depth-Anything-V2 Metric Indoor (Metric model ကမှ distance ကို meter နဲ့ ပြပေးတာပါ)
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device=0 if torch.cuda.is_available() else -1)

def get_distance_with_yolo(frame, calibration_factor=0.7):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # --- Step 1: Depth Estimation ---
    inputs = pipe.image_processor(images=img_pil, return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        outputs = pipe.model(**inputs)
        # Model output ကို မူရင်း image size အတိုင်း ပြန်ချဲ့တာပါ
        post_processed_output = pipe.image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(frame.shape[0], frame.shape[1])]
        )
        # တကယ့် meter value ပါတဲ့ depth map ကို ယူတာပါ
        depth_map = post_processed_output[0]["predicted_depth"].cpu().numpy()
        
        # Calibration factor ကို သုံးပြီး depth ကို ပြင်ဆင်တာပါ
        depth_map = depth_map * calibration_factor

    # --- Step 2: YOLO Detection ---
    results = yolo_model(frame, verbose=False)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names
        
        for box, cls in zip(boxes, classes):
            label = names[int(cls)]
            
            # Bottle ပဲ predict လုပ်မယ် - အခြား objects တွေကို skip လုပ်မယ်
            if label != "person":
                continue
            
            x1, y1, x2, y2 = map(int, box)
            
            # Center of the box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Boundary check (Image size ထက် မကျော်အောင်)
            cy = min(max(0, cy), depth_map.shape[0] - 1)
            cx = min(max(0, cx), depth_map.shape[1] - 1)
            
            # Bounding box အတွင်း ရှိ depth values တွေထဲက အနီးဆုံး distance ကို ယူခြင်း
            y1_clipped = min(max(0, y1), depth_map.shape[0] - 1)
            y2_clipped = min(max(0, y2), depth_map.shape[0] - 1)
            x1_clipped = min(max(0, x1), depth_map.shape[1] - 1)
            x2_clipped = min(max(0, x2), depth_map.shape[1] - 1)
            
            # Bounding box area ရဲ့ minimum depth ယူတာ (အနီးဆုံး ဒီစ္စန်း)
            box_depth = depth_map[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
            if box_depth.size > 0:
                distance_m = np.min(box_depth)
            else:
                distance_m = depth_map[cy, cx]
            
            distance_ft = distance_m * 3.28084  # Meter ကို Feet သို့ ပြောင်းလဲခြင်း
            feet = int(distance_ft)
            inches = (distance_ft - feet) * 12
            
            # Visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Distance ကို Feet နဲ့ Inches နဲ့ ပြခြင်း
            text = f"{label}: {feet}'{inches:.1f}\"" 
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame

# စမ်းသပ်ရန် Video သို့မဟုတ် Webcam ဖွင့်ခြင်း
cap = cv2.VideoCapture(0) # 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    output_frame = get_distance_with_yolo(frame)
    
    cv2.imshow("YOLO11 + Depth-Anything V2 Distance", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()