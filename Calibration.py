import cv2
import numpy as np
import os
import time
import json

def find_and_crop_chessboard_squares(output_dir="cropped_squares"):
    """
    Captures an image from the webcam at a specific resolution, finds a 
    chessboard pattern, crops each individual square, and saves the images
    and their pixel locations.

    Args:
        output_dir (str): The directory where cropped square images will be saved.
    """
    # --- 1. Setup and Webcam Capture ---
    print("Initializing webcam...")
    # Use 0 for the default webcam. Adding cv2.CAP_DSHOW can improve compatibility on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- SET CAMERA RESOLUTION ---
    # Set the desired width and height.
    # Note: The webcam must support the resolution you request.
    desired_width = 1920
    desired_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    print(f"Attempting to set resolution to {desired_width}x{desired_height}...")

    # Give the camera a moment to initialize
    time.sleep(2)

    print("Capturing image... Please show a chessboard to the camera.")
    ret, frame = cap.read()
    cap.release() 

    if not ret:
        print("Error: Could not capture image from webcam.")
        return

    # --- 2. Find Chessboard Corners ---
    print("Image captured successfully.")
    h, w, _ = frame.shape
    print(f"Captured image resolution: {w}x{h}")
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(1000)

    print("Searching for chessboard corners...")
    pattern_size = (7, 7)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if not ret:
        print("Chessboard not found. Make sure the entire board is visible and well-lit.")
        cv2.destroyAllWindows()
        return

    print("Chessboard corners found!")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # --- 3. Crop, Save Squares, and Record Data ---
    print("Cropping squares and recording data...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # List to hold the data for each square
    squares_data = []

    corner_0 = corners[0][0]
    corner_1 = corners[1][0]
    corner_7 = corners[7][0]

    h_step = corner_1 - corner_0
    v_step = corner_7 - corner_0

    top_left_coord = corner_0 - h_step - v_step

    for row in range(8):
        for col in range(8):
            pt1 = top_left_coord + row * v_step + col * h_step
            pt2 = pt1 + h_step + v_step
            
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x1 < x2 and y1 < y2:
                square = frame[y1:y2, x1:x2]
                
                file_name = f"square_{chr(ord('a')+col)}{8-row}.png"
                file_path = os.path.join(output_dir, file_name)
                cv2.imwrite(file_path, square)
                
                # Create a dictionary for the current square's data
                square_info = {
                    "file_name": file_name,
                    "top_left_corner": (x1, y1),
                    "bottom_right_corner": (x2, y2)
                }
                squares_data.append(square_info)

    # --- 4. Save Coordinate Data to JSON File ---
    json_output_path = os.path.join(output_dir, "square_locations.json")
    with open(json_output_path, 'w') as f:
        json.dump(squares_data, f, indent=4)

    print(f"\nSuccessfully cropped {len(squares_data)} squares.")
    print(f"Image files saved in the '{output_dir}' directory.")
    print(f"Pixel locations saved to '{json_output_path}'")
    
    cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
    cv2.imshow('Chessboard with Corners', frame)

    print("\nPress any key to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    find_and_crop_chessboard_squares()
