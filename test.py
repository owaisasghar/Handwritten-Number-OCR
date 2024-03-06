
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def cropImage(image, box):
    """Crop the image based on the bounding box and save the cropped image to a specified path.

    Args:
        image (np.ndarray): The original image.
        box (list): The bounding box coordinates as [x1, y1, x2, y2, x3, y3, x4, y4].
        save_path (str): The path (including filename) where the cropped image will be saved.

    Returns:
        np.ndarray: The cropped image region.
    """
    # Convert box points to a numpy array and calculate min and max coordinates
    box = np.array(box).reshape(-1, 2)
    min_x, min_y = np.min(box, axis=0).astype(int)
    max_x, max_y = np.max(box, axis=0).astype(int)

    # Use integer indices to slice the image
    cropped = image[min_y:max_y, min_x:max_x]

    # Save the cropped image to the specified path
    # cv2.imwrite( cropped)

    return cropped


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

import cv2
from PIL import Image
# Assuming 'processor' and 'model' are defined and initialized elsewhere in your code

def apply_trocr_and_extract_numbers(image_array):
    """Apply TrOCR on a cropped image array to extract only numbers, with custom replacements."""

    # Check if the image array is empty
    if image_array is None or image_array.size == 0:
        print("Empty or invalid image array.")
        return ""  # Return an empty string or handle the error as appropriate

    # Ensure the image is in RGB format
    if len(image_array.shape) == 2:  # Convert grayscale images to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_array)

    # Process the image with TrOCR
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    # Custom replacements: 'G' -> '6', 'D' -> '0', additional replacements as needed
    text = text.replace("G", "6").replace("D", "0").replace("S", "5").replace("O", "0").replace("C", "0").replace("I", "1").replace("Z", "2")

    return text

def draw_merged_boxes(image, merged_boxes, display_width=800):
    # Assuming 'image' is the image array on which to draw the boxes
    vis_image = image.copy()  # Create a copy of the image to draw on

    # Calculate the scaling factor to resize the image
    height, width = vis_image.shape[:2]
    scaling_factor = display_width / width

    # Resize image to fit on the screen
    vis_image = cv2.resize(vis_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    for box in merged_boxes:
        # Scale box coordinates
        scaled_box = [(int(x[0] * scaling_factor), int(x[1] * scaling_factor)) for x in box]

        # Extract the top-left and bottom-right points for cv2.rectangle
        top_left = scaled_box[0]  # Top-left corner
        bottom_right = scaled_box[2]  # Bottom-right corner

        # Draw the rectangle on the resized image
        cv2.rectangle(vis_image, top_left, bottom_right, (0, 255, 0), 2)  # Green box with thickness of 2

    # Display the resized image with bounding boxe
def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()


    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    
    # Calculate the average height of the bounding boxes to determine row grouping
    avg_height = np.mean([np.abs(b[0][1] - b[2][1]) for b in boxes])
    row_threshold = avg_height / 2  # Set a threshold to group boxes into rows

    # Group boxes by rows based on their y-coordinate
    rows = {}  # This dictionary will store bounding boxes grouped by rows
    for box in boxes:
        center_y = np.mean([point[1] for point in box])
        found_row = False
        for row_y in rows.keys():
            if abs(center_y - row_y) < row_threshold:
                rows[row_y].append(box)
                found_row = True
                break
        if not found_row:
            rows[center_y] = [box]

        
    import csv

    # Define the CSV file path
    csv_file_path = 'extracted_texts.csv'

    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['Row Y-Coordinate', 'Number', 'Price'])
        
        for row_y, row_boxes in rows.items():
            row_boxes = sorted(row_boxes, key=lambda b: np.min([p[0] for p in b]))
            
            row_text = []
            for box in row_boxes:
                cropped_image = cropImage(image, box)  # Your cropImage function
                text = apply_trocr_and_extract_numbers(cropped_image)  # Your text extraction function
                row_text.append(text)
                    
            # Combine text from all boxes in the row
            combined_text = ' '.join(row_text)
            print(f"Row at y={row_y}: {combined_text}")
            draw_merged_boxes(image, row_boxes, display_width=800)
            # Parsing combined_text to separate 'Number' and 'Price'
            parts = combined_text.split()
            print(parts)

            # data = [int(item) for item in parts if item.isdigit()]


            # # A function to pair the quantities with their prices
            # def pair_quantities_prices(data_list):
            #     return [(data_list[i], data_list[i + 1]) for i in range(0, len(data_list), 2)]

            # # Use the function to create the paired list
            # paired_data = pair_quantities_prices(data)

            # # Print out the paired data
            # for quantity, price in paired_data:
            #     print(f'Quantity: {quantity}, Price: {price}')

            # # Write the paired data to a CSV file.
            # with open('quantities_and_prices.csv', 'a', newline='') as csvfile:
            #     csvwriter = csv.writer(csvfile)
            #     # Writing the headers
            #     # csvwriter.writerow(['Quantity', 'Price'])
            #     # Writing the data
            #     csvwriter.writerows(paired_data)

            # print("Data has been written to 'quantities_and_prices.csv'")




        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))


        return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        # net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location=torch.device('cpu'))))

    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
        # net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location=torch.device('cpu'))))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))


