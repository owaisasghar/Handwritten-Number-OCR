import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2
import os
    # Initialize the processor and model for TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

def apply_trocr_and_extract_numbers(image_path):
    """Apply TrOCR on an image to extract only numbers, with custom replacements for 'G' and 'D'."""

    # Load the image and ensure it's in RGB
    image = cv2.imread(image_path)
    if len(image.shape) == 2:  # Convert grayscale images to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Process the image with TrOCR
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Custom replacements: 'G' -> '6', 'D' -> '0'
    text = text.replace("G", "6").replace("D", "0").replace("S","5").replace("O","0").replace("C","0").replace("I","1")

    # Extract numbers using regular expression
    numbers = ' '.join(re.findall(r'\d+', text))
    return text

def process_folder_for_numbers(folder_path):
    """Process all images in a folder, applying TrOCR to recognize and extract only numbers with custom replacements."""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # Include other image formats as needed
            image_path = os.path.join(folder_path, filename)
            numbers = apply_trocr_and_extract_numbers(image_path)
            print(f"Detected Numbers in {filename}: {numbers}")

# Specify the folder path containing your images
folder_path = 'cropped_images'
process_folder_for_numbers(folder_path)
