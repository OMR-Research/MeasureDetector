import os

import numpy as np
import argparse

import torch
from PIL import Image
from PIL.ImageDraw import ImageDraw

from StaveDetector.train_detection_refiner import DetectionRefinementModel
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor


def run_inference_for_single_image(image: Image.Image, model):
    image_tensor = ToTensor()(image)
    image_tensor = default_collate([image_tensor])
    prediction = model(image_tensor, [])[0]
    relative_center_x, relative_center_y, relative_width, relative_height = prediction
    image_width, image_height = image.size
    center_x, center_y = relative_center_x * image_width, relative_center_y * image_height
    width, height = relative_width * image_width, relative_height * image_height
    left = center_x - (width / 2)
    right = center_x + (width / 2)
    top = center_y - (height / 2)
    bottom = center_y + (height / 2)

    return [left, top, right, bottom]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs detection over input image given a trained detector.')
    parser.add_argument('--trained_model', dest='trained_model', type=str, required=True,
                        help='Path to the trained model.')
    parser.add_argument('--input_image', dest='input_image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_image', dest='output_image', type=str, default='detection.jpg',
                        help='Path to the output image.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DetectionRefinementModel()
    model.load_state_dict(torch.load(args.trained_model))

    image = Image.open(args.input_image).convert("RGB")  # type:Image.Image
    predicted_box_1, predicted_box_2 = run_inference_for_single_image(image, model)
    image_draw = ImageDraw(image)
    for predicted_box in [predicted_box_1, predicted_box_2]:
        int_box = [int(v) for v in predicted_box]
        image_draw.rectangle(int_box, outline='#008888', width=2)
    image.show()
