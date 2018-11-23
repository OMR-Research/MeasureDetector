import argparse
import json
from glob import glob
import os
import urllib.request

from lxml import etree
from tqdm import tqdm


def download_and_extract_dataset(dataset: str, base_url: str):
    for source in glob(f'data/{dataset}/*.xml'):
        base = os.path.splitext(source)[0]
        os.makedirs(base, exist_ok=True)

        print("Downloading dataset for " + base)

        xml = etree.parse(source).getroot()

        # Download pictures
        for graphic in tqdm(xml.xpath('//*[local-name()="graphic"]'), desc="Downloading images"):
            url = graphic.get('target')
            filename = os.path.basename(url)
            width = graphic.get('width')
            if os.path.exists(os.path.join(base, filename)):
                # print(f'Skipping  {os.path.join(base, filename)} because it has already been downloaded...')
                pass  # Skipping download, because it has been downloaded already
            else:
                # print(f'Downloading {url}...')
                urllib.request.urlretrieve(f"{base_url}/{url}?dw={width}&amp;mo=fit", os.path.join(base, filename))

        # Extract bar annotations
        for surface in tqdm(xml.xpath('//*[local-name()="surface"]'), desc="Creating json annotations"):
            image_path = os.path.join(base, surface[0].get('target').split('/')[-1])
            size = (int(surface[0].get('width')), int(surface[0].get('height')))

            json_path = os.path.splitext(image_path)[0] + '.json'

            system_measures = []
            for zone in surface.xpath('./*[local-name()="zone"][@type="measure"]'):
                left = int(zone.get('ulx'))
                top = int(zone.get('uly'))
                right = int(zone.get('lrx'))
                bottom = int(zone.get('lry'))

                data = {'left': left, 'top': top, 'right': right, 'bottom': bottom}
                system_measures.append(data)

            stave_measures = []
            staves = []
            with open(json_path, 'w') as file:
                json.dump({'width': size[0], 'height': size[1], 'system_measures': system_measures,
                           'stave_measures': stave_measures, 'staves': staves}, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloads a dataset from the Edirom system')
    parser.add_argument('-dataset', dest='dataset', type=str, required=True,
                        help='Must be either "Bargheer" or "FreischuetzDigital"')
    parser.add_argument('-url', dest='url', type=str, required=True, help='URL where to download the dataset from')
    args = parser.parse_args()

    if args.dataset not in ["Bargheer", "FreischuetzDigital"]:
        raise Exception("Invalid dataset specified. Must be either 'Bargheer' or 'FreischuetzDigital'")

    download_and_extract_dataset(args.dataset, args.url)
