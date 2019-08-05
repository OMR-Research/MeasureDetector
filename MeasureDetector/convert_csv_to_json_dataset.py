import argparse
import json
import os
from glob import glob

from PIL import Image
from tqdm import tqdm
import pandas as pd


def convert_annotations_to_one_json_file_per_image(dataset_directory: str):
    if len(glob(f'{dataset_directory}/**/*.txt', recursive=True)) == 0:
        print(f"Could not find annotation files in {dataset_directory} directory.")

    image_files = glob(f'{dataset_directory}/**/*.jpg', recursive=True)
    annotation_files = glob(f'{dataset_directory}/**/*.txt', recursive=True)
    assert len(image_files) == len(annotation_files)

    for image_file, annotation_file in tqdm(zip(image_files, annotation_files),
                                            desc="Converting annotations",
                                            total=len(image_files)):
        dataset_annotations = pd.read_csv(annotation_file, names=["left", "top", "right", "bottom", "class"])

        system_measures = []
        stave_measures = []
        staves = []
        engraving = "handwritten"

        for index, stave in dataset_annotations[dataset_annotations["class"] == "staff"].iterrows():
            left = stave["left"]
            top = stave["top"]
            right = stave["right"]
            bottom = stave["bottom"]

            data = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
            staves.append(data)

        json_filename = os.path.splitext(os.path.basename(image_file))[0] + ".json"
        image = Image.open(image_file)
        width, height = image.width, image.height
        json_path = os.path.join(dataset_directory, json_filename)
        with open(json_path, 'w') as file:
            json.dump(
                {'width': width, 'height': height, 'engraving': engraving, 'system_measures': system_measures,
                 'stave_measures': stave_measures, 'staves': staves}, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare single-file annotations. Seaches the given directory '
                                                 'recursively for *.txt and *.jpg files that will be extracted into'
                                                 'plain json annotation files, one file per image.'
                                                 'Assumes that the directory contains one txt-file with csv-annotations'
                                                 'in the format "left,top,right,bottom,class" per line.')
    parser.add_argument("--dataset_directory", type=str, default="data",
                        help="The directory, where the extracted dataset resides")

    flags = parser.parse_args()
    convert_annotations_to_one_json_file_per_image(flags.dataset_directory)
