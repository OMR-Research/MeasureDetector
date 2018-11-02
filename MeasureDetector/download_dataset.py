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

            bars = []
            for zone in surface.xpath('./*[local-name()="zone"][@type="measure"]'):
                ulx = int(zone.get('ulx'))
                uly = int(zone.get('uly'))
                lrx = int(zone.get('lrx'))
                lry = int(zone.get('lry'))
                width = lrx - ulx
                height = lry - uly
                x = ulx
                y = uly

                data = {'x': x, 'y': y, 'width': width, 'height': height}
                bars.append(data)

            with open(json_path, 'w') as file:
                json.dump({'width': size[0], 'height': size[1], 'bars': bars}, file)


if __name__ == "__main__":
    download_and_extract_dataset("FreischuetzDigital", "https://digilib.freischuetz-digital.de/Scaler")
    download_and_extract_dataset("Bargheer", "https://bargheer.edirom.de/Scaler")
