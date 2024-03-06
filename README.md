# Handwritten Number Detection

This project utilizes the CRAFT (Character Region Awareness for Text detection) model for detecting handwritten numbers and TROCR (Transformer-based Optical Character Recognition) for OCR to recognize these numbers from images.

## Installation

Ensure you have Python installed on your system. This project has been tested with Python 3.8. Before running the detection script, you need to install the required libraries. Run the following command to install them:

```bash
pip install torch==2.2.0 torchvision==0.17.0

## Code Run 
python check.py --trained_model=model.pth --test_folder=data --cuda=False


Make sure to adjust the paths and file names as per your project's structure and naming conventions. If your project does not include a script named `check.py`, you will need to replace it with the correct script name that performs the detection and OCR tasks.

