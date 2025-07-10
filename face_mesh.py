import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize MediaPipe Drawing Utilities (we'll only use it for its structure, not its drawing methods directly for the symbols)
mp_draw = mp.solutions.drawing_utils

# --- Personalized Greeting ---
GREETING_MESSAGE = "HELLO DEVANDLA RAJU"

# --- Load the Love Symbol Image ---
# IMPORTANT: Make sure 'blue_heart_small.png' is in the same directory as this script!
try:
    love_symbol_img = cv2.imread('blue_heart_small.png', cv2.IMREAD_UNCHANGED)
    if love_symbol_img is None:
        raise FileNotFoundError("blue_heart_small.png not found or could not be loaded.")

    # Check if the image has an alpha channel (for transparency)
    if love_symbol_img.shape[2] == 4:
        # Split the image into BGR channels and the alpha channel
        love_symbol_bgr = love_symbol_img[:, :, 0:3]
        love_symbol_alpha = love_symbol_img[:, :, 3]
    else:
        # If no alpha channel, assume it's fully opaque and create a full alpha mask
        love_symbol_bgr = love_symbol_img
        love_symbol_alpha = np.ones(love_symbol_img.shape[:2], dtype=np.uint8) * 255
    
    symbol_h, symbol_w = love_symbol_bgr.shape[:2]

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure 'blue_heart_small.png' is in the same directory as the script.")
    print("Defaulting to drawing small blue circles instead of love symbols.")
    love_symbol_img = None # Indicate that image loading failed

# --- Function to Overlay an Image with Alpha Channel ---
def overlay_alpha_image(background, overlay_bgr, overlay_alpha, x, y):
    """
    Overlays an image with an alpha channel onto a background image.
    x, y are the top-left coordinates for the overlay.
    """
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay_bgr.shape[:2]

    # Calculate region of interest (ROI)
    y1, y2 = max(0, y), min(bg_h, y + ol_h)
    x1, x2 = max(0, x), min(bg_w, x + ol_w)

    # Calculate overlay region considering image boundaries
    ol_y1, ol_y2 = max(0, -y), min(ol_h, bg_h - y)
    ol_x1, ol_x2 = max(0, -x), min(ol_w, bg_w - x)

    # Ensure valid regions
    if ol_y2 <= ol_y1 or ol_x2 <= ol_x1:
        return background # Overlay region is empty

    # Get the ROI from the background
    roi = background[y1:y2, x1:x2]

    # Get the corresponding part of the overlay image and alpha mask
    overlay_roi_bgr = overlay_bgr[ol_y1:ol_y2, ol_x1:ol_x2]
    overlay_roi_alpha = overlay_alpha[ol_y1:ol_y2, ol_x1:ol_x2]

    # Convert alpha to a float range [0, 1]
    alpha_factor = overlay_roi_alpha / 255.0
    alpha_factor_3ch = cv2.merge[alpha_factor, alpha_factor, alpha_factor]