import cv2
import pytesseract
import os

def apply_tesseract_ocr_to_folder(folder_path):
    # Specify the path to tesseract.exe on Windows, if necessary
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Add other file types if needed
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Convert image to grayscale for better OCR results
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Tesseract OCR to recognize only numbers
            custom_config = r'--oem 3 --psm 13 outputbase digits'
            text = pytesseract.image_to_string(gray_image, config=custom_config)

            print(f"Detected Numbers in {filename}: {text}")

# Example usage
folder_path = 'cropped_images'
apply_tesseract_ocr_to_folder(folder_path)
