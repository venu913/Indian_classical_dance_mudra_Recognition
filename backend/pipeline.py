import cv2
import numpy as np
import mediapipe as mp

def enhance_image(image_bgr):
    """
    DIP (Digital Image Processing) Enhancement for Low-End Cameras.
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting issues.
    2. Unsharp Masking for unblurring/sharpening hands from low-quality sensors.
    """
    # 1. Lighting Fix: Convert to LAB color space to apply CLAHE to the L (Lightness) channel
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    # 2. Unblur Fix: Unsharp Masking
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
    sharpened_bgr = cv2.addWeighted(enhanced_bgr, 1.5, gaussian, -0.5, 0)
    
    return sharpened_bgr

def contextual_crop(image_bgr, bbox, padding_ratio=0.30):
    """
    V2.0 Contextual Crop Algorithm
    Instead of tightly cropping on the YOLO bounding box, we expand the box by a specific 
    ratio (e.g. 30%). This guarantees MediaPipe never loses the wrist (Landmark 0) or fingertips.
    """
    x1, y1, x2, y2 = bbox
    H, W = image_bgr.shape[:2]
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)
    
    # Expand coordinates, clamped to image boundaries
    px1 = max(0, x1 - pad_x)
    py1 = max(0, y1 - pad_y)
    px2 = min(W, x2 + pad_x)
    py2 = min(H, y2 + pad_y)
    
    return image_bgr[py1:py2, px1:px2]

def extract_6ch_tensor(crop_bgr, mp_hands):
    """
    Given a perfectly context-padded crop, extract the 21 MediaPipe landmarks 
    and fuse them into a 6-channel tensor (RGB + Heatmaps).
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(crop_rgb)
    
    if not res.multi_hand_landmarks:
        return None  # Should be extremely rare now due to contextual_crop
        
    best_lm = max(res.multi_hand_landmarks, key=lambda hl: sum(lm.visibility for lm in hl.landmark))
    
    ch, cw = crop_rgb.shape[:2]
    # Draw connections
    skeleton_img = np.zeros((ch, cw, 3), dtype=np.uint8)
    
    # We use Mediapipe drawing utils to draw thick lines
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing.draw_landmarks(
        skeleton_img,
        best_lm,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    
    # Resize to exactly 224x224
    rgb_224 = cv2.resize(crop_rgb, (224, 224))
    skel_224 = cv2.resize(skeleton_img, (224, 224))
    
    # Concatenate along channel dimension -> (224, 224, 6)
    tensor_6ch = np.concatenate([rgb_224, skel_224], axis=-1)
    return tensor_6ch
