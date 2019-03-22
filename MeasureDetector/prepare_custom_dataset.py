import argparse
import json
import os
from glob import glob

from tqdm import tqdm


def convert_annotations_to_one_json_file_per_image(dataset_directory: str):
    if len(glob(f'{dataset_directory}/**/dataset.json', recursive=True)) == 0:
        print(f"Could not find json files in {dataset_directory} directory.")

    for dataset in tqdm(glob(f'{dataset_directory}/**/dataset.json', recursive=True), desc="Converting annotations"):
        with open(dataset, "r") as file:
            dataset_annotations = json.load(file)

        root_directory = dataset_annotations["root_dir"]

        for name, source in dataset_annotations["sources"].items():
            source_directory = source["root_dir"]

            for page in source["pages"]:
                system_measures = []
                width = page["width"]
                height = page["height"]

                for measure in page["annotations"]["measures"]:
                    left = int(measure["bbox"]["x"])
                    top = int(measure["bbox"]["y"])
                    right = int(measure["bbox"]["x"] + measure["bbox"]["width"])
                    bottom = int(measure["bbox"]["y"] + measure["bbox"]["height"])

                    data = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
                    system_measures.append(data)

                # Currently, the dataset only has system measure annotation, so we leave the other two types empty
                stave_measures = []
                staves = []
                json_filename = os.path.splitext(page["image"])[0] + ".json"
                json_path = os.path.join(dataset_directory, root_directory, source_directory, json_filename)
                with open(json_path, 'w') as file:
                    json.dump({'width': width, 'height': height, 'system_measures': system_measures,
                               'stave_measures': stave_measures, 'staves': staves}, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare single-file annotations. Seaches the given directory '
                                                 'recursively for dataset.json files that will be extracted into'
                                                 'plain json annotation files, one file per image.')
    parser.add_argument("--dataset_directory", type=str, default="data/weber",
                        help="The directory, where the extracted dataset resides")

    flags = parser.parse_args()
    convert_annotations_to_one_json_file_per_image(flags.dataset_directory)
