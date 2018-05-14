import datetime
import cv2
import threading
import time
import tensorflow as tf
import queue as Queue


class FPS:
    '''Class for FPS counter'''

    def __init__(self, interval):
        self.globStart = None
        self.globEnd = None
        self.globNumFrames = 0
        self.localStart = None
        self.localNumFrames = 0
        self.interval = interval
        self.currLocalElapsed = None
        self.first = False

    def start(self):
        self.globStart = datetime.datetime.now()
        self.localStart = self.globStart
        return self

    def stop(self):
        self.globEnd = datetime.datetime.now()

    def update(self):
        self.first = True
        curr_time = datetime.datetime.now()
        self.currLocalElapsed = (curr_time - self.localStart).total_seconds()
        self.globNumFrames += 1
        self.localNumFrames += 1
        if self.currLocalElapsed > self.interval:
            print('FPS: {}'.format(self.fpsLocal()))
            self.localNumFrames = 0
            self.localStart = curr_time

    def elapsed(self):
        return (self.globEnd - self.globStart).total_seconds()

    def fps(self):
        return self.globNumFrames / self.elapsed()

    def fpsLocal(self):
        if self.first:
            return round(self.localNumFrames / self.currLocalElapsed, 1)
        else:
            return 0.0


class WebcamVideoStream:
    '''Class for threaded webcam stream'''

    def __init__(self, src, width, height):
        self.frameCounter = 1
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(src)
        #self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        #self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        (self.grabbed, self.frame) = self.stream.read()
        self.threadStop = False
        #self.real_width = int(self.stream.get(3))
        #self.real_height = int(self.stream.get(4))
        print(
            '[INFO] Starting video stream with shape: {},{}'.format(
                width, height))

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until threadStop is true
        while True:
            if self.threadStop:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.frameCounter += 1

    def read(self):
        return self.frame

    def stop(self):
        self.threadStop = True

    def isActive(self):
        return self.stream.isOpened

    def resize(self):
        try:
            self.frame = cv2.resize(self.frame, (self.width, self.height))
        except:
            print('[ERROR] Can not resize video stream')


class SessionWorker():
    '''Class for threading tensorflow sessions'''

    def __init__(self, tag, graph, config):
        self.lock = threading.Lock()
        self.sessQueue = Queue.Queue()
        self.resultQueue = Queue.Queue()
        self.tag = tag
        t = threading.Thread(target=self.execution, args=(graph, config))
        t.setDaemon(True)
        t.start()
        return

    def execution(self, graph, config):
        self.threadRunning = True
        try:
            with tf.Session(graph=graph, config=config) as sess:
                while self.threadRunning:
                    while not self.sessQueue.empty():
                        q = self.sessQueue.get(block=False)
                        opts = q['opts']
                        feeds = q['feeds']
                        extras = q['extras']
                        if feeds is None:
                            results = sess.run(opts)
                        else:
                            results = sess.run(opts, feed_dict=feeds)
                        self.resultQueue.put(
                            {'results': results, 'extras': extras})
                        self.sessQueue.task_done()
                    time.sleep(0.005)
        except:
            #print('[ERROR] Multithreading failed')
            import traceback
            traceback.print_exc()
        self.stop()

    def stop(self):
        self.threadRunning = False
        with self.lock:
            while not self.sessQueue.empty():
                q = self.sessQueue.get(block=False)
                self.sessQueue.task_done()

    def sessEmpty(self):
        return self.sessQueue.empty()

    def getResultQueue(self):
        result = None
        if not self.resultQueue.empty():
            result = self.resultQueue.get(block=False)
            self.resultQueue.task_done()
        return result

    def insertSessQueue(self, opts, feeds=None, extras=None):
        self.sessQueue.put({"opts": opts, "feeds": feeds, "extras": extras})
