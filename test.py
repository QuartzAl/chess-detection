import cv2
import numpy as np

# Load the image
image = cv2.imread('test\\square_g3.png')

if image is None:
    print("Error: Could not read the image.")
else:
    # --- Define the color range for Black ---
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to RGB for color range detection
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([2, 2, 2]) # Adjust this upper bound for "near blacks"
    black_mask = cv2.inRange(image, lower_black, upper_black)

    # --- Define the color range for White ---
    lower_white = np.array([80, 0, 0]) # Adjust this lower bound for "near whites"
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    cv2.imwrite('white_mask.png', white_mask)  # Save the white mask for debugging
    cv2.imwrite('black_mask.png', black_mask)  # Save the black mask for debugging

    # --- Combine the black and white masks ---
    # We use bitwise_or to combine the two masks. A pixel will be selected if it's in the black range OR the white range.
    combined_mask = cv2.bitwise_or(black_mask, white_mask)

    # Invert the mask to select the pixels we want to KEEP
    inverse_mask = cv2.bitwise_not(combined_mask)

    # Convert the original image to BGRA format to include an alpha channel
    bgra_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    bgra_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel of the BGRA image using the inverted mask
    # Where the inverted mask is 0 (the colors to be removed), the alpha will be 0 (transparent)
    bgra_image[:, :, 3] = inverse_mask

    # Save the resulting image with a transparent background
    cv2.imwrite('result_bgr_transparent.png', bgra_image)
    print("Image with transparent blacks and whites saved as 'result_bgr_transparent.png'")

    # Optional: Display the images