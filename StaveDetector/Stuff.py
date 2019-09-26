import os
import shutil
from glob import glob
from random import sample

import pandas as pd
from tqdm import tqdm

detections = pd.read_csv("MeasureDetector/detection_output/detections.csv")

detections = detections[["image_name", "class_name"]]
grouped = detections.groupby("image_name").count()
series = pd.Series(grouped["class_name"])
filenames = grouped.index.tolist()
images_to_remove = set()
mapping = {}

for i in range(len(filenames)):
    image_name = filenames[i].split("_D-")[0]
    number_of_staves = series[i]
    if image_name in mapping:
        if number_of_staves != mapping[image_name]:
            images_to_remove.add(image_name)
    else:
        mapping[image_name] = number_of_staves

images_to_remove = list(images_to_remove)
images_to_remove.sort()
print(images_to_remove)

all_files = glob("MeasureDetector/detection_output/*.*")
all_files = glob("MeasureDetector/data/muscima_pp/v1.0/data/CVCMUSCIMA_MCA_Flat/*.*")
for image_to_remove in images_to_remove:
    for file in all_files:
        if image_to_remove in file:
            print("removing " + file)
            os.remove(file)

directory = r"D:\Dropbox\Stave Detection\MaskDetection\2.Iteration\masks_input\CVCMUSCIMA_2000"
## Delete json files that don't have images
json_files = glob(os.path.join(directory, "*.json"))
image_files = glob(os.path.join(directory, "*_annotated.png"))
print(len(json_files))
print(len(image_files))

i = 0
image_file_names = [os.path.splitext(os.path.basename(i))[0].replace("_annotated", "") for i in image_files]
for json_file in json_files:
    file = os.path.splitext(os.path.basename(json_file))[0]
    if file not in image_file_names:
        i += 1
        print("Removing {0} because the image does not exist".format(json_file))
        os.remove(json_file)

print(f"Deleted {i}/{len(json_files)} files")

## Delete images that don't have annotations
json_files = glob(os.path.join(directory, "*.json"))
image_files = glob(os.path.join(directory, "*.png"))
print(len(json_files))
print(len(image_files))

i = 0
json_file_names = [os.path.splitext(os.path.basename(i))[0] for i in json_files]
for image_file in image_files:
    file = os.path.splitext(os.path.basename(image_file))[0]
    if file not in json_file_names:
        i += 1
        print("Removing {0} because the json file does not exist".format(image_file))
        os.remove(image_file)

print(f"Deleted {i}/{len(image_files)} images")

# Remove images, that have already been served as ground-truth
all_annotated_images = glob(r"E:\Stave Detection\4. Iteration\input\deep_scores/*.png")
len(all_annotated_images)
all_checked_images = glob(r"E:\Stave Detection\4. Iteration\input\deep_scores/*.png")
len(all_checked_images)
unchecked_images = glob(r"E:\Dropbox\Stave Detection\deep_scores_detection_output/*.png")
len(unchecked_images)

all_annotated_images = [os.path.basename(a).replace("_detection", "") for a in all_annotated_images]
all_checked_images = [os.path.basename(a).replace("_detection", "") for a in all_checked_images]
all_verified_images = set(all_annotated_images).union(set(all_checked_images))
len(all_verified_images)

i = 0
for unchecked_image in unchecked_images:
    if os.path.basename(unchecked_image).replace("_detection", "") in all_verified_images:
        i += 1
        print("Removing {0} because it has already been checked".format(unchecked_image))
        os.remove(unchecked_image)

print("Removed {0}/{1} already checked images".format(i, len(unchecked_images)))

# Select a random sample of 2000 images from the existing dataset to not overwhelm the network
all_images = glob(r"E:\Stave Detection\CVCMUSCIMA_MCA_Flat\*.png")
random_samples = sample(all_images, 2000)
destination_directory = r"E:\Stave Detection\5. Iteration\input\CVCMUSCIMA_2000"
os.makedirs(destination_directory, exist_ok=True)
for random_sample in tqdm(random_samples):
    shutil.copy(random_sample, destination_directory)
    shutil.copy(random_sample.replace(".png", ".json"), destination_directory)
