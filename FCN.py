import logging
from PIL import Image
import numpy as np
import os

from keras.models import Model, load_model, model_from_json, Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, concatenate
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam


class FCN16:
    def build(weight_path = None, is_training = False):
        # shape = (h, w, ch), the dimension of batch_size is omitted ??
        inputs = Input(shape=(None, None, 3))

        # build vgg16 without top(fully conneted layers)
        vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
        
        # build those deconv layers

        # deconv, dilation_rate ??
        if is_training:
            deconv1_1 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation,
                                        kernel_initializer = Constant(FCN16.bilinear_upsample_weights(2, 512, 512)),
                                        name = 'deconv1_1')(vgg16.output)
        else:
            deconv1_1 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation, name = 'deconv1_1')(vgg16.output)
        deconv1_2 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv1_2')(deconv1_1)
        deconv1_3 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv1_3')(deconv1_2)
        deconv1_4 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv1_4')(deconv1_3)

        if is_training:
            drop1_5 = Dropout(FCN16.dropped_rate, name = 'drop1_5')(deconv1_4)
            deconv2_1 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation,
                                        kernel_initializer = Constant(FCN16.bilinear_upsample_weights(2, FCN16.n_filters[0], FCN16.n_filters[0])),
                                        name = 'deconv2_1')(drop1_5)
        else:
            deconv2_1 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation, name = 'deconv2_1')(deconv1_4)
        deconv2_2 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv2_2')(deconv2_1)
        deconv2_3 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv2_3')(deconv2_2)
        deconv2_4 = Conv2DTranspose(filters = FCN16.n_filters[1], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv2_4')(deconv2_3)
        
        if is_training:
            drop2_5 = Dropout(FCN16.dropped_rate, name = 'drop2_5')(deconv2_4)
            deconv3_1 = Conv2DTranspose(filters = FCN16.n_filters[1], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation,
                                        kernel_initializer = Constant(FCN16.bilinear_upsample_weights(2, FCN16.n_filters[1], FCN16.n_filters[1])),
                                        name = 'deconv3_1')(drop2_5)
        else:
            deconv3_1 = Conv2DTranspose(filters = FCN16.n_filters[1], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation, name = 'deconv3_1')(deconv2_4)
        deconv3_2 = Conv2DTranspose(filters = FCN16.n_filters[1], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv3_2')(deconv3_1)
        deconv3_3 = Conv2DTranspose(filters = FCN16.n_filters[1], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv3_3')(deconv3_2)
        deconv3_4 = Conv2DTranspose(filters = FCN16.n_filters[2], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv3_4')(deconv3_3)

        if is_training:
            drop3_5 = Dropout(FCN16.dropped_rate, name = 'drop3_5')(deconv3_4)
            deconv4_1 = Conv2DTranspose(filters = FCN16.n_filters[2], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation,
                                        kernel_initializer = Constant(FCN16.bilinear_upsample_weights(2, FCN16.n_filters[2], FCN16.n_filters[2])),
                                        name = 'deconv4_1')(drop3_5)
        else:
            deconv4_1 = Conv2DTranspose(filters = FCN16.n_filters[2], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation, name = 'deconv4_1')(deconv3_4)
        deconv4_2 = Conv2DTranspose(filters = FCN16.n_filters[2], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv4_2')(deconv4_1)
        deconv4_3 = Conv2DTranspose(filters = FCN16.n_filters[3], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv4_3')(deconv4_2)

        if is_training:
            drop4_4 = Dropout(FCN16.dropped_rate, name = 'drop4_4')(deconv4_3)
            deconv5_1 = Conv2DTranspose(filters = FCN16.n_filters[3], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation,
                                        kernel_initializer = Constant(FCN16.bilinear_upsample_weights(2, FCN16.n_filters[3], FCN16.n_filters[3])),
                                        name = 'deconv5_1')(drop4_4)
        else:
            deconv5_1 = Conv2DTranspose(filters = FCN16.n_filters[3], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                        activation = FCN16.deconv_activation, name = 'deconv5_1')(deconv4_3)
        deconv5_2 = Conv2DTranspose(filters = FCN16.n_filters[3], kernel_size = (4, 4), strides = (1, 1), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv5_2')(deconv5_1)
        conv5_3 = Conv2D(filters = FCN16.n_filters[4], kernel_size = (1, 1), strides = (1, 1), padding = 'same',
                                    activation = 'tanh', use_bias = True, name = 'conv5_3')(deconv5_2)

        model = Model(inputs = inputs, outputs = conv5_3)

        for layer in model.layers[:15]:
            layer.trainable = False

        # load the previous model
        if weight_path and os.path.isfile(weight_path):
            logging.info('model exists')
            
            model.load_weights(weight_path, by_name = True)

        # output the layer info of model
        model.summary()

        return model

    def post_processing(imgs, mode = 'tf'):
        if mode == 'tf':
            imgs += 1.
            imgs *= 127.5

            imgs = imgs.astype(np.uint8)
        else:
            logging.error('unexpected mode = %s' % mode)
            exit()
        
        return imgs

    def predict_and_save(model, img, out_path):
        pred = model.predict(img)[0]
        
        # post processing
        pred = np.squeeze(pred, axis = 2)
        pred = FCN16.post_processing(pred)

        out_img = Image.fromarray(pred, mode = 'L')

        # save the out_img
        out_img.save(out_path)
    
    def test_dir(model, gen, out_dir):
        for data, name, th in gen:
            out_video_dir = os.path.join(out_dir, name)
            if not os.path.exists(out_video_dir):
                os.mkdir(out_video_dir)

            ori_data = data[0]
            s_data = data[1]

            preds = model.predict_on_batch(ori_data)

            # post processing
            preds = np.squeeze(preds, axis = 3)
            preds = FCN16.post_processing(preds)

            for pred in preds:
                out_img = Image.fromarray(pred, mode = 'L')
  
                out_path = '%s/%.5d.png' % (out_video_dir, th)
                th += 1

                # save the out_img
                out_img.save(out_path)

    # Bilinear interpolation (reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/upsampling.py)
    def bilinear_upsample_weights(factor, out_n_class, in_n_class):
        filter_size = factor*2 - factor%2
        factor = (filter_size + 1) // 2

        if filter_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        
        og = np.ogrid[:filter_size, :filter_size]
        upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

        weights = np.zeros((filter_size, filter_size, out_n_class, in_n_class), dtype = np.float32)

        n = out_n_class if out_n_class < in_n_class else in_n_class
        for i in range(n):
            weights[:, :, i, i] = upsample_kernel

        return weights


    # some neural network setting
    n_filters = [128, 64, 32, 8, 1]
    deconv_activation = 'relu'
    dropped_rate = 0.4
