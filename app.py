import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import imutils
from flask import Flask, jsonify, request
import uuid
import time

app = Flask(__name__)

print('[INFO] Load neural net (GPU)...')
options = {
    'model': 'cfg/yolov2.cfg',
    'load': 'weights/yolov2.weights',
    'threshold': 0.25,
    'gpu': 0.8
}
tfnet = TFNet(options)

@app.route('/api/detections', methods=['POST'])
def save_detection():

    # read frame from the request
    frame = np.array(request.json['frame'])

    # get detected objects
    obj_detected = request.json['obj_detected']

    # generate unique temp filename to save
    tmp_folder = 'detections'
    dt = time.strftime('%Y-%m-%d_%H-%M-%S')
    objects = "_".join(obj_detected)
    tmp_filename = '{}__{}__{}.jpg'.format(dt, objects, str(uuid.uuid4())[:4])
    cv2.imwrite('{}/{}'.format(tmp_folder, tmp_filename), frame)

    # return ok
    return jsonify({'status': 'ok'})

@app.route('/api/predict', methods=['POST'])
def make_predict():

    # read frame from the request
    frame = np.array(request.json['frame'])
    
    # initialise empty results (in case prediction fails)
    new_results = []

    try:
        # generate unique temp filename to save,
        # for some reason function return_predict fails
        # with an OpenCV C++ "resize" error when
        # frame is passed in directly to it
        tmp_folder = 'tmp_images'
        tmp_filename = '{}.jpg'.format(str(uuid.uuid4()))
        cv2.imwrite('{}/{}'.format(tmp_folder, tmp_filename), frame)

        # read image and pass to the network
        frame = cv2.imread('{}/{}'.format(tmp_folder, tmp_filename))
        results = tfnet.return_predict(frame)

        # convert confidence to float and append to results
        new_results = []
        for r in results:
            r_new = r.copy()
            r_new['confidence'] = r_new['confidence'].astype(float)
            new_results.append(r_new)

        # clean-up/remove tmp image
        os.remove('{}/{}'.format(tmp_folder, tmp_filename))
    except Exception as e:
        print(e)
        print('[ERROR] Could not generate predictions (return_predict failed)')
    return jsonify({'results': new_results})

if __name__ == '__main__':
    print('[INFO] Start Flask /api/predict route...')
    app.run()