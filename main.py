import logging
import sys
import os
import time
from PIL import Image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD, Adam
from keras.models import load_model, model_from_json
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K
import json

from FCN import FCN16


# logging setting
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(name)-12s %(levelname)-8s\n%(message)s',
                    datefmt = '%Y-%m-%d %H:%M',
                    handlers = [logging.FileHandler('main.log', 'w', 'utf-8'), ])

console = logging.StreamHandler(sys.stdout)
# console.setLevel(logging.DEBUG)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('').addHandler(console)

# data
data_dir = 'data'
video_dir = os.path.join(data_dir, 'video')
training_dir = os.path.join(video_dir, 'training')
validation_dir = os.path.join(video_dir, 'validation')
test_dir = os.path.join(video_dir, 'test')
pred_dir = os.path.join(video_dir, 'prediction')
ori_dir = 'frame'
s_dir = 'saliency'
model_log_dir = os.path.join(data_dir, 'model_log')
# model_path = os.path.join(data_dir, 'model.h5')
# model_json_path = os.path.join(data_dir, 'model_json.json')
model_weight_path = os.path.join(data_dir, 'model_weight.h5')

# neural network setting
learning_rate = 0.00001
loss = 'mse'
# valid_rate = 0.1
# n_epoch = 10
out_epoch = 5
# in_epoch = 10000
in_epoch = 5
batch_size = 2


def load_image(path):
    if os.path.isfile(path):
        img_org = Image.open(path)
        w, h = img_org.size

        # ensure that the size of img is divisible by 32
        img = img_org.resize(((w // 32) * 32, (h // 32) * 32))

        del img_org, w, h

        img = np.array(img, dtype = np.float32)

        if len(img.shape) != 3:
            img = np.expand_dims(img, axis = 2)

        # in tf mode, it will scale pixels between -1 and 1 sample-wise
        img = preprocess_input(img, mode = 'tf')

        return img
    else:
        logging.error('load_image error: path is not a file')
        exit()

def generate_arrays_from_file(d, ori_idr, label_dir, batch_size = batch_size, is_infinite = True):
    while 1:
        for name in os.listdir(d):
            video_path = os.path.join(d, name)
            if os.path.isdir(video_path):
                ori_path = os.path.join(video_path, ori_idr)
                label_path = os.path.join(video_path, label_dir)

                batch_counter = 0
                xs = []
                ys = []

                # there is a 'Output' dir in the ori_path
                # for img_name in os.listdir(ori_path):
                for img_name in os.listdir(label_path):
                    ori_img_path = os.path.join(ori_path, img_name)
                    label_img_path = os.path.join(label_path, img_name)

                    x = load_image(ori_img_path)
                    y = load_image(label_img_path)
                    
                    xs.append(x)
                    ys.append(y)
                    
                    del ori_img_path, label_img_path, x, y

                    batch_counter += 1
                    if batch_counter >= batch_size:
                        xs = np.array(xs, dtype = np.float32)
                        ys = np.array(ys, dtype = np.float32)
                        
                        yield(xs, ys)

                        batch_counter = 0

                        del xs, ys
                        xs = []
                        ys = []
                
                if len(ys) != 0:
                    xs = np.array(xs, dtype = np.float32)
                    ys = np.array(ys, dtype = np.float32)

                    yield(xs, ys)

                del ori_path, label_path, batch_counter, xs, ys
            else:
                logging.erro('the path %s is not a dir' % video_path)
                exit(0)
                
            del video_path
        
        if not is_infinite:
            break

def generate_from_a_dir(d, ori_dir, label_dir, batch_size = batch_size, is_infinite = True):
    while 1:
        for name in os.listdir(d):
            video_path = os.path.join(d, name)
            if os.path.isdir(video_path):
                ori_path = os.path.join(video_path, ori_dir)
                label_path = os.path.join(video_path, label_dir)

                batch_counter = 0
                th = 0
                xs = []
                ys = []

                # there is a 'Output' dir in the ori_path
                # for img_name in os.listdir(ori_path):
                for img_name in os.listdir(label_path):
                    ori_img_path = os.path.join(ori_path, img_name)
                    label_img_path = os.path.join(label_path, img_name)

                    x = load_image(ori_img_path)
                    y = load_image(label_img_path)
                    
                    xs.append(x)
                    ys.append(y)

                    del ori_img_path, label_img_path, x, y

                    batch_counter += 1
                    if batch_counter >= batch_size:
                        xs = np.array(xs, dtype = np.float32)
                        ys = np.array(ys, dtype = np.float32)
                        
                        yield (xs, ys), name, th

                        th += batch_counter
                        batch_counter = 0

                        del xs, ys
                        xs = []
                        ys = []
                
                if len(ys) != 0:
                    xs = np.array(xs, dtype = np.float32)
                    ys = np.array(ys, dtype = np.float32)

                    yield (xs, ys), name, th

                del ori_path, label_path, batch_counter, th, xs, ys
            else:
                logging.error('the path %s is not a dir' % video_path)
                exit()

            del video_path
        
        if not is_infinite:
            break

def count_data(d, label_dir):
    count = 0

    for name in os.listdir(d):
        video_path = os.path.join(d, name)
        if os.path.isdir(video_path):
            label_path = os.path.join(video_path, label_dir)

            for img_name in os.listdir(label_path):
                count += 1
            
            del label_path
        else:
            logging.error('error in count_data: video_path is not a dir = %s' % video_path)
            exit()

        del video_path
    
    return count


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            t0 = time.time()

            # count total count of training data
            n_training = count_data(training_dir, s_dir)
            n_validation = count_data(validation_dir, s_dir)
            logging.info('n_training = %d' % n_training)
            logging.info('n_validation = %d' % n_validation)

            model = FCN16.build(model_weight_path, is_training = True)
            logging.info('build model completed')

            # learning rate is very important!?
            adam = Adam(lr = learning_rate)

            model.compile(loss = loss, optimizer = adam)

            for epoch in range(out_epoch):
                logging.info('out epoch = %d' % epoch)

                model.fit_generator(
                    generate_arrays_from_file(training_dir, ori_dir, s_dir),
                    steps_per_epoch = n_training / batch_size,
                    epochs = in_epoch,
                    validation_data = generate_arrays_from_file(validation_dir, ori_dir, s_dir),
                    validation_steps = n_validation / batch_size,
                    callbacks = [TensorBoard(log_dir = model_log_dir, batch_size = batch_size)])

                # save all model info
                # model.save(model_path)
                
                # save the weight of model
                model.save_weights(model_weight_path)

            logging.info('training time = %s' % (time.time() - t0))
        elif sys.argv[1] == 'predict' and len(sys.argv) >= 3:
            t0 = time.time()

            logging.info('prediction time = %s' % (time.time() - t0))
        elif sys.argv[1] == 'test':
            t0 = time.time()

            if len(sys.argv) == 2:
                n_test = count_data(test_dir, s_dir)
                logging.info('n_test = %d' % n_test)

                model = FCN16.build(model_weight_path)
                logging.info('build model completed')

                FCN16.test_dir(model, generate_from_a_dir(test_dir, ori_dir, s_dir, is_infinite = False), pred_dir)
            else:
                logging.error('unexpected sys.argv len in test')

            logging.info('test time = %s' % (time.time() - t0))
    else:
        logging.error('unexpected sys.argv len = %d' % len(sys.argv))