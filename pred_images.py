import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from PIL import ImageFont
import imutils

print('[INFO] Load model...')
options = {
    'model': 'cfg/yolov2.cfg',
    'load': 'weights/yolov2.weights',
    'threshold': 0.3,
    'gpu': 0.8
}
tfnet = TFNet(options)

print('[INFO] Start inference...')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
in_path = './images'
out_path = './out'
for filename in [f for f in os.listdir(in_path) if f.endswith(".jpg")]:
    frame = cv2.imread("{}/{}".format(in_path, filename))

    cv2.imwrite('test_img.jpg', frame)

    frame = imutils.resize(frame, width=400)
    print('shape', frame.shape, frame[0].shape, frame[0])
    break

    print('[INFO] WIP on: {}...'.format(filename))

    if frame is None:
        print('Could not load file {} from {}'.format(filename, in_path))
        continue

    results = tfnet.return_predict(frame)
    for color, result in zip(colors, results):
        start_y = result['topleft']['y']
        tl = (result['topleft']['x'], start_y - 15 if start_y - 15 > 15 else start_y + 15)
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        frame = cv2.rectangle(frame, tl, br, color, 2)
        frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (98,250,60), 2)

    cv2.imwrite("{}/{}".format(out_path, filename), frame)

print('[INFO] Done. Close TF session')
tfnet.sess.close()