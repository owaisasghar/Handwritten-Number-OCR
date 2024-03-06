
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import imgproc 
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import csv
import json
import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
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
parser.add_argument('--low_text', default=0.25, type=float, help='text low-bound score')
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
    # Convert box to a NumPy array and reshape to 4x2
    box = np.array(box).reshape(-1, 2)

    # Compute the axis-aligned bounding box of the box
    rect = cv2.minAreaRect(box)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Crop the image using the axis-aligned bounding box
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(image, M, (width, height))

    return cropped


import numpy as np

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

    # Display the resized image with bounding boxes
    cv2.imshow('Merged Boxes', vis_image)
    cv2.waitKey(0)  # Wait for a key press to exit
    cv2.destroyAllWindows()  # Close the image window


def merge_close_horizontal_boxes(boxes, closeness_threshold):
    if not boxes:
        return boxes

    # Sort boxes by the x-coordinate of their left edge
    boxes.sort(key=lambda x: x[0][0])
    merged_boxes = [boxes[0]]

    for current_box in boxes[1:]:
        previous_box = merged_boxes[-1]
        # Calculate horizontal distance between the current box and the previous box
        distance = current_box[0][0] - previous_box[2][0]

        # Check if the current box is close enough to the previous box to merge
        if distance <= closeness_threshold:
            # Merge the current box with the previous box
            top_left_y = min(previous_box[0][1], current_box[0][1])
            top_left_x = min(previous_box[0][0], current_box[0][0])
            bottom_right_y = max(previous_box[2][1], current_box[2][1])
            bottom_right_x = max(previous_box[2][0], current_box[2][0])
            # Update the last element in merged_boxes with the new merged box
            merged_boxes[-1] = [(top_left_x, top_left_y), (bottom_right_x, top_left_y), (bottom_right_x, bottom_right_y), (top_left_x, bottom_right_y)]
        else:
            # If not close enough, simply add the current box to the list of merged boxes
            merged_boxes.append(current_box)

    return merged_boxes
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

def apply_trocr_and_extract_numbers(image_array):

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
    text = text.replace("G", "6").replace("D", "0").replace("S", "5").replace("O", "0").replace("C", "0").replace("I", "1").replace("Z", "2").replace(".","").replace("T","7").replace("Y","4").replace("L","1").replace("-"," ")

    return text


def combine_numbers(numbers):
    combined = []
    skip_next = False

    for i in range(len(numbers)):
        if skip_next:
            # Skip this iteration because the previous number was combined with this one
            skip_next = False
            continue
        
        if i < len(numbers) - 1 and (len(str(numbers[i])) == 1 or len(str(numbers[i])) == 2):
            # Combine this number with the next one if it's one or two digits long
            combined_number = int(f"{numbers[i]}{numbers[i + 1]}")
            combined.append(combined_number)
            skip_next = True  # Skip the next number since it was combined with the current one
        else:
            combined.append(numbers[i])
    
    return combined

# A function to pair the quantities with their prices
def pair_quantities_prices(data_list):
    return [(data_list[i], data_list[i + 1]) for i in range(0, len(data_list) - 1, 2)]


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

    # Assuming definitions of cropImage, apply_trocr_and_extract_numbers, combine_numbers, pair_quantities_prices, and merge_close_horizontal_boxes are available
    csv_file_path = 'quantities_and_prices.csv'
    json_file_path = 'quantities_and_prices.json'
    all_data = []

    # Set to keep track of processed image names
    processed_images = set()

        # Load existing JSON data to append new data instead of overwriting
    try:
        with open(json_file_path, mode='r') as jsonfile:
            all_data = json.load(jsonfile)
    except (FileNotFoundError, json.JSONDecodeError):
        all_data = []
    # Open the CSV file in append mode to add data for each image processed
    with open(csv_file_path, mode='a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Before processing, load existing image names from the CSV to avoid duplication
        with open(csv_file_path, mode='r', newline='') as readfile:
            csvreader = csv.reader(readfile)
            for row in csvreader:
                if row:  # Check if row is not empty
                    processed_images.add(row[0])  # Assuming the first column is the image name

        for image_path in image_list:  # Assuming image_list is defined elsewhere and contains all image paths
            image_name = os.path.basename(image_path)  # Extract the filename from the image_path
            
            # Check if the image name has already been processed
            is_loaded = False

            if image_name not in processed_images:
                # Optionally, write the name of the image being processed as a header or comment
                csvwriter.writerow(['Image Name', image_name])
                
                # Check if it's the first time loading
                if not is_loaded:
                    # Optionally, write a header row for each image section (remove if not needed)
                    csvwriter.writerow(['Quantity', 'Price'])
                    is_loaded = True  # Update the flag after loading once

        # Assuming 'rows' is updated for each image in the image_list
        for row_y, row_boxes in rows.items():
            row_boxes = sorted(row_boxes, key=lambda b: np.min([p[0] for p in b]))
            closeness_threshold = 10  # Example threshold

            merged_boxes = merge_close_horizontal_boxes(row_boxes, closeness_threshold)
        
            row_text = []
            for box in merged_boxes:
                cropped_image = cropImage(image, box)  # Make sure this function correctly crops the image
                text = apply_trocr_and_extract_numbers(cropped_image)  # Extract text
            
                row_text.append(text)
            
        
        # Combine text from all boxes in the row
            combined_text = ' '.join(row_text)
            
            # Extract and combine numbers from the text
            numbers = [int(item) for item in combined_text.split() if item.isdigit()]
            combined_numbers = combine_numbers(numbers)
            
            # Pair the quantities with their prices
            paired_data = pair_quantities_prices(combined_numbers)
            # Print and write the paired data
            for quantity, price in paired_data:
                print(f'Quantity: {quantity}, Price: {price}')
                csvwriter.writerow([quantity, price])

                # Add data to all_data list for JSON output
                all_data.append({"image_name": image_name, "quantity": quantity, "price": price})
                
        # Optionally, add a blank row for better separation between images in CSV
        csvwriter.writerow([])  # Add a blank row for separation

        # After processing all images, write the complete data to a JSON file
        with open(json_file_path, mode='w') as jsonfile:
            json.dump(all_data, jsonfile, indent=4)

        print("Data has been written to 'quantities_and_prices.csv'")
        print(f"Quantities and prices from all processed images have been written to '{json_file_path}'")

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

