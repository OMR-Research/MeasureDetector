import json
import os

import requests
from PIL import Image
from PIL.ImageDraw import ImageDraw

image_path = 'IMSLP454435-PMLP738602-Il_tempio_d_amore_Sinfonia-0011.jpg'

# Get bounding boxes for all measures
with open(image_path, 'rb') as image:
    response = requests.post('http://localhost:8080/upload', files={'image': image})
measures = json.loads(response.content.decode('utf-8'))['measures']
print(measures)

# Draw boxes in copy of source image
image = Image.open(image_path).convert('RGBA')
temp = Image.new('RGBA', image.size)
draw = ImageDraw(temp)

for m in measures:
    draw.rectangle([int(m['ulx']), int(m['uly']), int(m['lrx']), int(m['lry'])], fill='#00FFFF1B')
for m in measures:
    draw.rectangle([int(m['ulx']), int(m['uly']), int(m['lrx']), int(m['lry'])], outline='#008888', width=2)

result_image = Image.alpha_composite(image, temp).convert('RGB')

basename, ext = os.path.splitext(image_path)
result_path = basename + '_bboxes' + ext
result_image.save(result_path)
