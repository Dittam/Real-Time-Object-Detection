# Protobuf Compile command
# os.system('protoc object_detection/protos/*.proto --python_out=.')

import numpy as np
import os
import tensorflow as tf
import copy
import time
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2
from TensorFlowDetectionAPI.utils import label_map_util
from TensorFlowDetectionAPI.utils import visualization_utils as vis_util
from extraFunctions import FPS, WebcamVideoStream, SessionWorker


# -----INIT CONFIG PARAMS-----
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
    categoryIndex = label_map_util.create_category_index(categories)
    return categoryIndex


def nodeName(n):
    '''(TF Node) -> str
    get the name of node in tensorflow graph
    '''
    if n.startswith('^'):
        result = n[1:]
    else:
        result = n.split(':')[0]
    return result


# helper function for split model

def loadFrozenModel():
    print('[INFO] Loading frozen model into memory...')
    # load a frozen Model and split it into GPU and CPU graphs
    inputGraph = tf.Graph()
    with tf.Session(graph=inputGraph):
        shape = 7326 if ssdShape == 600 else 1917

        score = tf.placeholder(tf.float32, shape=(
            None, shape, numClasses), name='Postprocessor/convert_scores')
        expand = tf.placeholder(tf.float32, shape=(
            None, shape, 1, 4), name='Postprocessor/ExpandDims_1')
        for node in inputGraph.as_graph_def().node:
            if node.name == 'Postprocessor/convert_scores':
                scoreDef = node
            if node.name == 'Postprocessor/ExpandDims_1':
                expandDef = node

        detectionGraph = tf.Graph()
        with detectionGraph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(modelPath, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                dest_nodes = ['Postprocessor/convert_scores',
                              'Postprocessor/ExpandDims_1']

                edges = {}
                name2node = {}
                node2seq = {}
                seq = 0
                for node in od_graph_def.node:
                    n = nodeName(node.name)
                    name2node[n] = node
                    edges[n] = [nodeName(x) for x in node.input]
                    node2seq[n] = seq
                    seq += 1
                for d in dest_nodes:
                    if d not in name2node:
                        print('[INFO] Node {} not in graph'.format(str(d)))

                keepNodes = set()
                visitNodes = dest_nodes[:]

                while len(visitNodes) > 0:
                    n = visitNodes[0]
                    del visitNodes[0]
                    if n not in keepNodes:
                        keepNodes.add(n)
                        visitNodes += edges[n]

                keepNodesList = sorted(
                    list(keepNodes), key=lambda n: node2seq[n])
                nodesToRemove = set()

                for n in node2seq:
                    if n not in keepNodesList:
                        nodesToRemove.add(n)
                nodesToRemoveList = sorted(
                    list(nodesToRemove), key=lambda n: node2seq[n])

                keep = graph_pb2.GraphDef()
                for n in keepNodesList:
                    keep.node.extend([copy.deepcopy(name2node[n])])

                remove = graph_pb2.GraphDef()
                remove.node.extend([scoreDef])
                remove.node.extend([expandDef])
                for n in nodesToRemoveList:
                    remove.node.extend([copy.deepcopy(name2node[n])])

                with tf.device('/gpu:0'):
                    tf.import_graph_def(keep, name='')
                with tf.device('/cpu:0'):
                    tf.import_graph_def(remove, name='')

        return detectionGraph, score, expand


def detection(detectionGraph, categoryIndex, score, expand):
    print('[INFO] Building Graph...')
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
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

            score_out = detectionGraph.get_tensor_by_name(
                'Postprocessor/convert_scores:0')
            expand_out = detectionGraph.get_tensor_by_name(
                'Postprocessor/ExpandDims_1:0')
            score_in = detectionGraph.get_tensor_by_name(
                'Postprocessor/convert_scores_1:0')
            expand_in = detectionGraph.get_tensor_by_name(
                'Postprocessor/ExpandDims_1_1:0')
            # Threading
            gpu_worker = SessionWorker('GPU', detectionGraph, config)
            cpu_worker = SessionWorker('CPU', detectionGraph, config)
            gpu_opts = [score_out, expand_out]
            cpu_opts = [detection_boxes, detection_scores,
                        detection_classes, num_detections]
            gpu_counter = 0
            cpu_counter = 0

            # -----Start Video Stream-----
            fps = FPS(fpsInterval).start()
            videoStream = WebcamVideoStream(
                videoInput, width, height).start()
            counts = 0
            print('[INFO] Press q to Exit')
            print('[INFO] Starting Detection...')
            while videoStream.isActive():
                # -----actual Detection-----
                image = videoStream.read()

                # split model in seperate gpu and cpu session threads
                if gpu_worker.sessEmpty():
                    # read video frame, expand dimensions and convert to rgb
                    image_expanded = np.expand_dims(
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                    # insert tensors into new queue
                    gpu_feeds = {image_tensor: image_expanded}
                    if visualize:
                        gpu_extras = image  # for visualization frame
                    else:
                        gpu_extras = None
                    gpu_worker.insertSessQueue(
                        gpu_opts, gpu_feeds, gpu_extras)
                g = gpu_worker.getResultQueue()
                if g is None:
                    # if gpu thread has no output queue
                    gpu_counter += 1
                else:
                    # if gpu thread has output queue.
                    gpu_counter = 0
                    score, expand, image = g['results'][
                        0], g['results'][1], g['extras']
                    if cpu_worker.sessEmpty():
                        # if cpu thread has no next put new queue
                        # else drop gpu queue
                        cpu_feeds = {
                            score_in: score, expand_in: expand}
                        cpu_extras = image
                        cpu_worker.insertSessQueue(
                            cpu_opts, cpu_feeds, cpu_extras)
                c = cpu_worker.getResultQueue()
                if c is None:
                    # cpu thread has no output queue
                    cpu_counter += 1
                    time.sleep(0.005)
                    continue  # If CPU result has not been set yet, no fps update
                else:
                    cpu_counter = 0
                    boxes, scores, classes, num, image = c['results'][0], c[
                        'results'][1], c['results'][2], c['results'][3], c['extras']

                # -----Visualization-----
                if visualize:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        categoryIndex,
                        use_normalized_coordinates=True,
                        line_thickness=3)
                    if visFPS:
                        cv2.putText(image, 'fps: {}'.format(fps.fpsLocal()), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

                cv2.imshow('object_detection', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to exit
                    break

                fps.update()
                counts += 1

    # close thread and video stream
    gpu_worker.stop()
    cpu_worker.stop()
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
