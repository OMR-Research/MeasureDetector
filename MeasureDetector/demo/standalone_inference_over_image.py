import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.ImageDraw import ImageDraw

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections',
                'detection_boxes',
                'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            return output_dict


def load_detection_graph(path_to_checkpoint):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performs detection over input image given a trained detector.')
    parser.add_argument('input_image', type=str, default="IMSLP454437-PMLP738602-Il_tempio_d_amore_Scene2-0002.jpg",
                        help='Path to the input image.')
    parser.add_argument('--detection_inference_graph', type=str,
                        default="2019-04-24_faster-rcnn_inception-resnet-v2.pb",
                        help='Path to the frozen inference graph.')

    parser.add_argument('--output_result', type=str, default="output_detections.json",
                        help='Path to the output file, that will contain a list of measures as JSON file')
    args = parser.parse_args()
    input_image_path = args.input_image
    basename, ext = os.path.splitext(input_image_path)
    output_image = basename + '_bboxes' + ext

    # Uncomment the next line on Windows to run the inference on the CPU, even though a GPU is available
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    detection_graph = load_detection_graph(args.detection_inference_graph)

    image = Image.open(input_image_path).convert("RGB")  # type: Image.Image
    image_np = np.array(image)
    image = image.convert("RGBA")
    overlay = Image.new('RGBA', image.size)
    image_draw = ImageDraw(overlay)
    (image_width, image_height) = image.size

    output_dict = run_inference_for_single_image(image_np, detection_graph)
    measures = []

    for idx in range(output_dict['num_detections']):
        if output_dict['detection_scores'][idx] > 0.5:

            y1, x1, y2, x2 = output_dict['detection_boxes'][idx]

            y1 = y1 * image_height
            y2 = y2 * image_height
            x1 = x1 * image_width
            x2 = x2 * image_width

            measures.append({
                'left': x1,
                'top': y1,
                'right': x2,
                'bottom': y2
            })

            if output_image is not None:
                image_draw.rectangle([int(x1), int(y1), int(x2), int(y2)], fill='#00FFFF1B')
                image_draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline='#008888', width=2)

        else:
            break

    if output_image is not None:
        result_image = Image.alpha_composite(image, overlay).convert('RGB')
        result_image.save(output_image)

    with open(args.output_result, "w") as output_file:
        json.dump(measures, output_file)
