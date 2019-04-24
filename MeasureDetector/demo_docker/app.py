import io

import hug
import numpy as np
import tensorflow as tf
from PIL import Image

# Initialize graph
detection_graph = tf.Graph()
detection_graph.as_default()
od_graph_def = tf.GraphDef()
with tf.gfile.GFile('model.pb', 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
sess = tf.Session()


def infer(image):
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

    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict


@hug.post('/upload')
def detect_measures(body):
    """Takes an image file and returns measure bounding boxes as JSON"""

    image = Image.open(io.BytesIO(body['image'])).convert("RGB")
    (image_width, image_height) = image.size
    image_np = np.array(image.getdata()).reshape((image_height, image_width, 3)).astype(np.uint8)

    output_dict = infer(image_np)
    measures = []

    for idx in range(output_dict['num_detections']):
        if output_dict['detection_classes'][idx] == 1 and output_dict['detection_scores'][idx] > 0.5:
            y1, x1, y2, x2 = output_dict['detection_boxes'][idx]

            y1 = y1 * image_height
            y2 = y2 * image_height
            x1 = x1 * image_width
            x2 = x2 * image_width

            measures.append({
                'ulx': x1,
                'uly': y1,
                'lrx': x2,
                'lry': y2
            })
        else:
            break

    return {'measures': measures}
