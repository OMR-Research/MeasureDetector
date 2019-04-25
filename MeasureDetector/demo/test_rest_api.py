import argparse
import json
import os

import requests
from PIL import Image
from PIL.ImageDraw import ImageDraw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs detection over input image given a trained detector.')
    parser.add_argument('--input_image', type=str, default='IMSLP454435-PMLP738602-Il_tempio_d_amore_Sinfonia-0011.jpg',
                        help='Path to the input image.')
    args = parser.parse_args()
    input_image_path = args.input_image
    basename, ext = os.path.splitext(input_image_path)

    # Get bounding boxes for all measures
    with open(input_image_path, 'rb') as image:
        response = requests.post('http://localhost:8080/upload', files={'image': image})
    measures = json.loads(response.content.decode('utf-8'))['measures']
    print(measures)

    # Draw boxes in copy of source image
    image = Image.open(input_image_path).convert('RGBA') # type: Image.Image
    overlay = Image.new('RGBA', image.size)
    image_draw = ImageDraw(overlay)

    for m in measures:
        image_draw.rectangle([int(m['left']), int(m['top']), int(m['right']), int(m['bottom'])], fill='#00FFFF1B')
    for m in measures:
        image_draw.rectangle([int(m['left']), int(m['top']), int(m['right']), int(m['bottom'])], outline='#008888', width=2)

    result_image = Image.alpha_composite(image, overlay).convert('RGB')

    basename, ext = os.path.splitext(input_image_path)
    result_path = basename + '_bboxes' + ext
    result_image.save(result_path)
