# Protobuf Compile command
# os.system('protoc object_detection/protos/*.proto --python_out=.')

import numpy as np
import os
import tensorflow as tf
import copy
import yaml
import time
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2
from TensorFlowDetectionAPI.utils import label_map_util
from TensorFlowDetectionAPI.utils import visualization_utils as vis_util
from extraFunctions import FPS, WebcamVideoStream, SessionWorker


# -----INITIALIZE CONFIG PARAMS-----
videoInput = 0
visualize = True
visFPS = False
height = 300
width = 500
fpsInterval = 5
allowMemGrth = True
modelName = 'ssd_mobilenet_v11_coco'
modelPath = 'models/ssd_mobilenet_v11_coco/frozen_inference_graph.pb'
labelPath = 'TensorFlowDetectionAPI/data/mscoco_label_map.pbtxt'
numClasses = 90
logDevice = False
ssdShape = 300


# Download Model form TF's Model Zoo
def download_model():
    model_file = modelName + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    if not os.path.isfile(modelPath):
        print('[INFO] Model not found. Downloading...')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('[INFO] Model found')


def load_labelmap():
    print('[INFO] Loading label map...')
    label_map = label_map_util.load_labelmap(labelPath)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=numClasses, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def nodeName(n):
    if n.startswith('^'):
        result = n[1:]
    else:
        result = n.split(':')[0]
    return result


# helper function for split model

def loadFrozenModel():
    print('[INFO] Loading frozen model into memory...')
    detectionGraph = tf.Graph()
    with detectionGraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(modelPath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detectionGraph, None, None


def detection(detectionGraph, category_index, score, expand):
    print('[INFO] Building Graph...')
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=logDevice)
    config.gpu_options.allow_growth = allowMemGrth
    curFrames = 0
    with detectionGraph.as_default():
        with tf.Session(graph=detectionGraph, config=config) as sess:
            # Define Input and Ouput tensors
            image_tensor = detectionGraph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detectionGraph.get_tensor_by_name(
                'detection_boxes:0')
            detection_scores = detectionGraph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detectionGraph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detectionGraph.get_tensor_by_name(
                'num_detections:0')

            # -----Start Video Stream-----
            fps = FPS(fpsInterval).start()
            videoStream = WebcamVideoStream(
                videoInput, width, height).start()
            print('[INFO] Press q to Exit')
            print('[INFO] Starting Detection...')

            while videoStream.isActive():
                # -----actual Detection-----
                image = videoStream.read()
                image_expanded = np.expand_dims(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                boxes, scores, classes, num = sess.run(
                    [detection_boxes, detection_scores,
                        detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

                # -----Visualization-----
                if visualize:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=3)
                    if visFPS:
                        cv2.putText(image, 'fps: {}'.format(fps.fpsLocal()), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                cv2.imshow('object_detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to exit
                    break
                fps.update()
    # close thread and video stream

    fps.stop()
    videoStream.stop()
    cv2.destroyAllWindows()
    print('[INFO] elapsed time (total): {:.2f}s'.format(fps.elapsed()))
    print('[INFO] Average FPS: {:.2f}'.format(fps.fps()))


if __name__ == '__main__':
    download_model()
    graph, score, expand = loadFrozenModel()
    category = load_labelmap()
    detection(graph, category, score, expand)
