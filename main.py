import logging
import sys
import os
import time
from PIL import Image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

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
test_dir = os.path.join(video_dir, 'test')
pred_dir = os.path.join(video_dir, 'prediction')
ori_dir = 'frame'
s_dir = 'saliency'
model_log_dir = os.path.join(data_dir, 'model_log')
model_path = os.path.join(data_dir, 'model.h5')

# neural network setting
learning_rate = 0.0001
loss = 'mse'
# n_epoch = 10
out_epoch = 10
# in_epoch = 10000
in_epoch = 2
batch_size = 8


def load_image(path):
    if os.path.isfile(path):
        img_org = Image.open(path)
        w, h = img_org.size

        # ensure that the size of img is divisible by 16
        img = img_org.resize(((w // 16) * 16, (h // 16) * 16))
        # img = np.asarray(img, dtype = np.float32)
        img = np.array(img, dtype = np.float32)

        # x = np.expand_dims(img, axis=0)

        if len(img.shape) != 3:
            img = np.expand_dims(img, axis = 2)
            # x /= 255.0

        # in tf mode, it will scale pixels between -1 and 1 sample-wise
        x = preprocess_input(img, mode = 'tf')

        return x
    else:
        logging.error('load_image error: path is not a file')
        exit()

def generate_arrays_from_file(d, img_dir, label_dir, batch_size = batch_size, is_infinite = True):
    while 1:
        for name in os.listdir(d):
            video_path = os.path.join(d, name)
            if os.path.isdir(video_path):
                ori_path = os.path.join(video_path, img_dir)
                s_path = os.path.join(video_path, label_dir)

                batch_counter = 0
                xs = []
                ys = []

                # there is a 'Output' dir in the ori_path
                # for img_name in os.listdir(ori_path):
                for img_name in os.listdir(s_path):
                    ori_img_path = os.path.join(ori_path, img_name)
                    s_img_path = os.path.join(s_path, img_name)

                    x = load_image(ori_img_path)
                    y = load_image(s_img_path)
                    
                    xs.append(x)
                    ys.append(y)

                    batch_counter += 1
                    if batch_counter >= batch_size:
                        xs = np.array(xs, dtype = np.float32)
                        ys = np.array(ys, dtype = np.float32)
                        
                        yield(xs, ys)

                        batch_counter = 0

                        del xs, ys
                        xs = []
                        ys = []
        
        if not is_infinite:
            break

def generate_from_a_dir(d, img_dir, label_dir, batch_size = batch_size, is_infinite = True):
    while 1:
        for name in os.listdir(d):
            video_path = os.path.join(d, name)
            if os.path.isdir(video_path):
                ori_path = os.path.join(video_path, img_dir)
                s_path = os.path.join(video_path, label_dir)

                batch_counter = 0
                th = 0
                xs = []
                ys = []

                # there is a 'Output' dir in the ori_path
                # for img_name in os.listdir(ori_path):
                for img_name in os.listdir(s_path):
                    ori_img_path = os.path.join(ori_path, img_name)
                    s_img_path = os.path.join(s_path, img_name)

                    x = load_image(ori_img_path)
                    y = load_image(s_img_path)
                    
                    xs.append(x)
                    ys.append(y)

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
                # batch_counter = 0

                del xs, ys
                # xs = []
                # ys = []

        
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
        else:
            logging.error('error in count_data: video_path is not a dir = %s' % video_path)
    
    return count


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'train':
            t0 = time.time()

            # count total count of training data
            n_training = count_data(training_dir, s_dir)
            # n_test = count_data(test_dir, s_dir)

            logging.info('n_training = %d' % n_training)
            
            # # create datagen
            # train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
            # test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

            # train_generator = train_datagen.flow_from_directory()

            model = FCN16.build(model_path)
            logging.info('build model completed')

            # model.compile(loss="binary_crossentropy", optimizer='sgd')

            # learning rate is very important!?
            # # sgd = SGD(lr = 0.01, momentum = 0.9, decay = 1e-6, nesterov = True)
            adam = Adam(lr = learning_rate)

            # # model.compile(loss = 'mse', optimizer = 'sgd')
            # # model.compile(loss = 'mse', optimizer = sgd)
            model.compile(loss = 'mse', optimizer = adam)

            # outputs = []

            # for layer in model.layers[15:]:
            #     outputs.append(K.function([model.layers[0].input], [layer.output]))

            for epoch in range(out_epoch):
                logging.info('out epoch = %d' % epoch)

                model.fit_generator(
                    generate_arrays_from_file(training_dir, ori_dir, s_dir),
                    steps_per_epoch = n_training / batch_size, 
                    epochs = in_epoch,
                    callbacks = [TensorBoard(log_dir = model_log_dir, batch_size = batch_size)])

                # in_path = os.path.join(video_dir, '104_conv_L', ori_dir, '00000.png')
                # # s_in_path = os.path.join(video_dir, '104_conv_L', s_dir, '00000.png')
                # out_path = os.path.join(data_dir, 'prediction', '%d_00000.png' % (epoch))

                # logging.info(get_19_output([load_image(in_path)])[0].shape)
                # logging.info(get_19_output([load_image(in_path)])[0])
                # logging.info('------------------------------------')
                # logging.info(get_20_output([load_image(in_path)])[0].shape)
                # logging.info(get_20_output([load_image(in_path)])[0])

                # for i in range(len(outputs)):
                #     logging.info('layer = %d' % (i + 15))
                #     logging.info(outputs[i]([load_image(in_path)])[0].shape)
                #     logging.info(outputs[i]([load_image(in_path)])[0])
                #     logging.info('------------------------------------')

                # FCN16.predict_and_save(model, np.expand_dims(load_image(in_path), axis = 0), out_path)
                model.save(model_path)

            # model.save(model_path)

            logging.info('training time = %s' % (time.time() - t0))
        elif sys.argv[1] == 'predict' and len(sys.argv) >= 3:
            t0 = time.time()

            # # load img data
            # imgs = np.asarray([np.asarray(Image.open(sys.argv[2], 'r').convert('RGB'))])

            # predicted_y = predict(imgs)
            # # logging.info(predicted_y)

            

            logging.info('prediction time = %s' % (time.time() - t0))
        elif sys.argv[1] == 'test':
            t0 = time.time()

            if len(sys.argv) == 2:
                # load img data
                n_test = count_data(test_dir, s_dir)

                logging.info('n_test = %d' % n_test)

                model = FCN16.build(model_path)
                logging.info('build model completed')

                FCN16.test_dir(model, generate_from_a_dir(test_dir, ori_dir, s_dir, is_infinite = False), pred_dir)
            else:
                logging.error('unexpected sys.argv len in test')

            # # load saliency data
            # s_imgs = np.asarray([np.asarray(Image.open(sys.argv[3], 'r').convert('L'))])
            # s_imgs = np.expand_dims(s_imgs, axis = len(s_imgs.shape))

            # predicted_y = predict(imgs, s_imgs)
            # # logging.info(predicted_y)
            # # logging.info(type(predicted_y))
            # # logging.info(predicted_y.shape)
            # # logging.info(predicted_y[0].shape)
            # predicted_y = predicted_y.reshape([1, 360, 640])
            # # logging.info(predicted_y.shape)
            # with open('temp.txt', 'w', encoding = 'utf-8') as f:
            #     for y in range(len(predicted_y[0][0])):
            #         for x in range(len(predicted_y[0])):
            #             f.write('%f, ' % predicted_y[0][x][y])

            #         f.write('\n')

            # result_y = Image.fromarray(predicted_y[0], mode = 'L')
            # # result_y.show()
            # result_y.save('result_y.png')

            logging.info('test time = %s' % (time.time() - t0))
    else:
        logging.error('unexpected sys.argv len = %d' % len(sys.argv))