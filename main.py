import cv2
import numpy as np
import os
import time
import json

def compare_chessboard_states(
    original_data_dir="cropped_squares", 
    comparison_output_dir="difference_squares"
    ):
    """
    Loads a previously saved chessboard configuration, captures a new image,
    and calculates the difference for each square to detect changes.

    Args:
        original_data_dir (str): Directory containing the original cropped squares
                                 and the 'square_locations.json' file.
        comparison_output_dir (str): Directory where the difference images
                                     will be saved.
    """
    # --- 1. Read from the JSON file ---
    json_path = os.path.join(original_data_dir, "square_locations.json")
    print(f"Reading board layout from: {json_path}")

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        print("Please run the first script ('chessboard_cropper') to generate the baseline images and data.")
        return

    with open(json_path, 'r') as f:
        squares_data = json.load(f)
    
    squares_data.sort(key=lambda x: x["file_name"])  # Sort by file name for consistency
    
    print("Successfully loaded square locations.")

    # --- 2. Take a picture from the webcam ---
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # IMPORTANT: Set the same resolution as the original capture script
    # to ensure the coordinates from the JSON file align correctly.
    desired_width = 1920
    desired_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    print(f"Attempting to set resolution to {desired_width}x{desired_height}...")
    time.sleep(2)

    print("Capturing new image... Please present the chessboard.")
    ret, new_frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not capture new image.")
        return

    h, w, _ = new_frame.shape
    print(f"Captured new image with resolution: {w}x{h}")
    cv2.imshow("Newly Captured Image", new_frame)
    cv2.waitKey(1000)

    # --- 3, 4, & 5. Crop, Subtract, and Save ---
    print("Processing differences...")
    
    if not os.path.exists(comparison_output_dir):
        os.makedirs(comparison_output_dir)
        print(f"Created output directory: {comparison_output_dir}")

    images = []

    for i, square_info in enumerate(squares_data):
        file_name = square_info["file_name"]
        top_left = square_info["top_left_corner"]
        bottom_right = square_info["bottom_right_corner"]
        
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Load the original square image (the baseline)
        original_square_path = os.path.join(original_data_dir, file_name)
        if not os.path.exists(original_square_path):
            print(f"Warning: Original square image not found: {original_square_path}. Skipping.")
            continue
            
        original_square = cv2.imread(original_square_path)
        
        # Crop the corresponding square from the NEW webcam frame
        new_square = new_frame[y1:y2, x1:x2]

        # Ensure dimensions match before subtraction. They should if the camera
        # resolution is consistent.
        if original_square.shape != new_square.shape:
            print(f"Warning: Dimension mismatch for {file_name}. Resizing new square.")
            new_square = cv2.resize(new_square, (original_square.shape[1], original_square.shape[0]))
        
        column = -1
        row = "z"

        for i, char in enumerate(file_name):
            if char.isdigit():
                number = int(char)
                row = file_name[i-1]
        
        
        
        
        # Calculate the absolute difference between the two squares
        difference = cv2.absdiff(original_square, new_square)
        
        # To make the difference more visible, we can convert to grayscale and threshold
        diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, threshold_diff = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        mask = cv2.cvtColor(threshold_diff, cv2.COLOR_GRAY2BGR)
        mask_float = mask.astype("float32") / 255.0
        new_square_float = new_square.astype("float32") / 255.0
        masked = cv2.multiply(new_square_float, mask_float)
        masked = (masked * 255).astype("uint8")


        # if (ord(row) - ord('a')) % 2 == 1:
        #     if number % 2 == 1:
        #         difference = cv2.bitwise_not(difference)
        # else:
        #     if number % 2 == 0:
        #         difference = cv2.bitwise_not(difference)

        lab_image = cv2.cvtColor(difference, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        merged_channels = cv2.merge([enhanced_l, a_channel, b_channel])
        final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)


        
        image = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)  # Convert to RGB for color range detection
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([2, 2, 2]) # Adjust this upper bound for "near blacks"
        black_mask = cv2.inRange(image, lower_black, upper_black)
        lower_white = np.array([80, 0, 0]) # Adjust this lower bound for "near whites"
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(image, lower_white, upper_white)
        combined_mask = cv2.bitwise_or(black_mask, white_mask)
        inverse_mask = cv2.bitwise_not(combined_mask)
        final_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2BGRA)
        final_image[:, :, 3] = inverse_mask
        transparent_mask = (final_image[:, :, 3] == 0)
        final_image[transparent_mask, 0:3] = 0

        gray = cv2.cvtColor(final_image, cv2.COLOR_BGRA2GRAY)
        _, binary_image = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((4,4), np.uint8)
        opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        mask = cv2.cvtColor(opened_image, cv2.COLOR_GRAY2BGR)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGRA2BGR)
        final_image = cv2.multiply(final_image.astype("float32") / 255.0, mask.astype("float32") / 255.0)


        image = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)  # Convert to RGB for color range detection
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([2, 2, 2]) # Adjust this upper bound for "near blacks"
        black_mask = cv2.inRange(image, lower_black, upper_black)

        # --- Define the color range for White ---
        lower_white = np.array([80, 0, 0]) # Adjust this lower bound for "near whites"
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(image, lower_white, upper_white)

        combined_mask = cv2.bitwise_or(black_mask, white_mask)
        inverse_mask = cv2.bitwise_not(combined_mask)
        final_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2BGRA)
        final_image[:, :, 3] = inverse_mask



        # Save the resulting difference image in the new folder
        output_file_name = f"diff_{file_name}"
        out_dir = "test" 
        output_path = os.path.join(comparison_output_dir, output_file_name)
        output_path2 = os.path.join(out_dir, f"{file_name}")
        print("Saving difference image to:", output_path)
        print("Saving masked image to:", output_path2)
        cv2.imwrite(output_path2, masked)
        cv2.imwrite(output_path, final_image)

    print(f"\nComparison complete. Difference images saved in '{comparison_output_dir}'.")
    cv2.destroyAllWindows()





if __name__ == '__main__':
    # Make sure the 'cropped_squares' folder from the previous script exists
    # in the same directory as this script.
    compare_chessboard_states()
