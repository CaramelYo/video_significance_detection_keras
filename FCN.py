import logging
from PIL import Image
import numpy as np
# import copy
import os

from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Conv2DTranspose
# from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam


class FCN16:
    def build(path = None, is_training = True):
        # load the previous model
        if path and os.path.isfile(path):
            logging.info('model exists')
            
            model = load_model(path)
            return model

        # shape = (h, w, ch), the dimension of batch_size is omitted ??
        inputs = Input(shape=(None, None, 3))

        # build vgg16 without top(fully conneted layers)
        vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

        # x = Conv2D(filters=nb_classes, 
        #            kernel_size=(1, 1))(vgg16.output)
        # x = Conv2DTranspose(filters=nb_classes, 
        #                     kernel_size=(64, 64),
        #                     strides=(32, 32),
        #                     padding='same',
        #                     activation='sigmoid',
        #                     kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(x)


        # build those deconv layers

        # deconv, dilation_rate ??
        # deconv1_1 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
        #                             activation = FCN16.deconv_activation,
        #                             kernel_initializer = Constant(bilinear_upsample_weights(2, FCN16.n_filters[0], FCN16.n_filters[0])),
        #                             name = 'deconv1_1')(vgg16.output)

        deconv1_1 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv1_1')(vgg16.output)

        # deconv1_2 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_1)
        # deconv1_3 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_2)
        # deconv1_4 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_3)
        
        # deconv1_4 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_1)

        # deconv1_4 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation)(deconv1_1)
        
        # deconv2_1 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_4)

        # deconv2_1 = Conv2DTranspose(filters = FCN16.n_filters[0], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = FCN16.deconv_activation)(deconv1_1)

        deconv2_1 = Conv2DTranspose(filters = FCN16.n_filters[1], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv2_1')(deconv1_1)

        # deconv2_2 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv2_1)
        # deconv2_3 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv2_2)
        # deconv2_4 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 512)))(deconv2_3)
        
        # deconv2_4 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 512)))(deconv2_1)

        # deconv2_4 = Conv2DTranspose(filters = FCN16.n_filters[1], kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation)(deconv2_1)
        
        # deconv3_1 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 256)))(deconv2_4)

        # deconv3_1 = Conv2DTranspose(filters = FCN16.n_filters[1], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = FCN16.deconv_activation)(deconv2_4)

        deconv3_1 = Conv2DTranspose(filters = FCN16.n_filters[2], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv3_1')(deconv2_1)

        # deconv3_2 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 256)))(deconv3_1)
        # deconv3_3 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 256)))(deconv3_2)
        # deconv3_4 = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 128, 256)))(deconv3_3)

        # deconv3_4 = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 128, 256)))(deconv3_1)

        # deconv3_4 = Conv2DTranspose(filters = FCN16.n_filters[2], kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation)(deconv3_1)

        # deconv4_1 = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 128, 128)))(deconv3_4)

        # deconv4_1 = Conv2DTranspose(filters = FCN16.n_filters[2], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = FCN16.deconv_activation)(deconv3_4)

        deconv4_1 = Conv2DTranspose(filters = FCN16.n_filters[3], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv4_1')(deconv3_1)

        # deconv4_2 = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 128, 128)))(deconv4_1)
        # deconv4_3 = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 64, 128)))(deconv4_2)
        
        # deconv4_3 = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 64, 128)))(deconv4_1)

        # deconv4_3 = Conv2DTranspose(filters = FCN16.n_filters[3], kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation)(deconv4_1)

        # deconv5_1 = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 64, 64)))(deconv4_3)

        # deconv5_1 = Conv2DTranspose(filters = FCN16.n_filters[3], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = FCN16.deconv_activation)(deconv4_3)

        deconv5_1 = Conv2DTranspose(filters = FCN16.n_filters[4], kernel_size = (4, 4), strides = (2, 2), padding = 'same',
                                    activation = FCN16.deconv_activation, name = 'deconv5_1')(deconv4_1)

        # deconv5_2 = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = FCN16.deconv_activation,
        #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 64, 64)))(deconv5_1)
        # conv5_3 = Conv2D(filters = 1, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'sigmoid', use_bias = False)(deconv5_2)

        conv5_3 = Conv2D(filters = FCN16.n_filters[5], kernel_size = (1, 1), strides = (1, 1), padding = 'same',
                            activation = 'tanh', use_bias = True, name = 'conv5_3')(deconv5_1)
        # conv5_3 = Conv2D(filters = 1, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'tanh', use_bias = False)(deconv5_1)

        model = Model(inputs = inputs, outputs = conv5_3)

        for layer in model.layers[:15]:
            layer.trainable = False

        # output the layer info of model
        model.summary()

        return model

    def predict_and_save(model, img, out_path):
        pred = model.predict(img)[0]
        pred = np.squeeze(pred, axis = 2)

        # from range (0, 1) to range(0, 255)
        # pred *= 255.0
        # from range (-1, 1) to range(0, 255)
        pred += 1.
        pred *= 127.5

        # logging.info('original type of pred = %s' % pred.dtype)

        # logging.info('shape = %s' % str(pred.shape))
        # logging.info('pred[45][512] = ' + str(pred[45][512]))

        pred = pred.astype(np.uint8)

        # logging.info('after type of pred = %s' % pred.dtype)

        # logging.info('shape = %s' % str(pred.shape))
        # logging.info('pred[45][512] = ' + str(pred[45][512]))

        # out_img = Image.fromarray(pred, mode='P')
        out_img = Image.fromarray(pred, mode = 'L')

        # save the out_img
        # palette_im = Image.open('palette.png')
        # img.palette = copy.copy(palette_im.palette)
        out_img.save(out_path)
    
    def test_dir(model, gen, out_dir):
        # gen_len = sum(1 for d in gen)
        # logging.info('gen_len = %d' % gen_len)

        for data, name, th in gen:
            logging.info('name = %s' % name)
            out_video_dir = os.path.join(out_dir, name)
            logging.info('out_video_dir = %s' % out_video_dir)
            if not os.path.exists(out_video_dir):
                os.mkdir(out_video_dir)

            ori_data = data[0]
            s_data = data[1]

            preds = model.predict_on_batch(ori_data)

            logging.info('shape of preds = %s' % str(preds.shape))

            preds = np.squeeze(preds, axis = 3)

            # from range (-1, 1) to range(0, 255)
            preds += 1.
            preds *= 127.5

            preds = preds.astype(np.uint8)

            for pred in preds:
                out_img = Image.fromarray(pred, mode = 'L')
  
                out_path = '%s/%d.png' % (out_video_dir, th)
                logging.info('out_path = %s' % out_path)
                th += 1

                # save the out_img
                out_img.save(out_path)

    # Bilinear interpolation (reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/upsampling.py)
    # def bilinear_upsample_weights(factor, number_of_classes):
    def bilinear_upsample_weights(factor, out_n_class, in_n_class):
        filter_size = factor*2 - factor%2
        factor = (filter_size + 1) // 2

        if filter_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        
        og = np.ogrid[:filter_size, :filter_size]
        upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

        # weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
        #                    dtype=np.float32)
        weights = np.zeros((filter_size, filter_size, out_n_class, in_n_class), dtype = np.float32)

        # for i in range(number_of_classes):
        n = out_n_class if out_n_class < in_n_class else in_n_class
        for i in range(n):
            weights[:, :, i, i] = upsample_kernel

        return weights
    
    # some neural network setting
    n_filters = [32, 16, 8, 4, 2, 1]
    deconv_activation = 'relu'






# # Bilinear interpolation (reference: https://github.com/warmspringwinds/tf-image-segmentation/blob/master/tf_image_segmentation/utils/upsampling.py)
# # def bilinear_upsample_weights(factor, number_of_classes):
# def bilinear_upsample_weights(factor, out_n_class, in_n_class):
#     filter_size = factor*2 - factor%2
#     factor = (filter_size + 1) // 2
#     if filter_size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:filter_size, :filter_size]
#     upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
#     # weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
#     #                    dtype=np.float32)

#     weights = np.zeros((filter_size, filter_size, out_n_class, in_n_class), dtype = np.float32)

#     # for i in range(number_of_classes):
    
#     n = out_n_class if out_n_class < in_n_class else in_n_class
#     for i in range(n):
#         weights[:, :, i, i] = upsample_kernel

#     return weights

# # def fcn_32s():
# def fcn_16s():
#     # shape = (h, w, ch), the dimension of batch_size is omitted ??
#     inputs = Input(shape=(None, None, 3))
#     # inputs = Input(shape = (None, None, None, 3))

#     n_filters = [32, 16, 8, 4, 2, 1]

#     vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)

#     # x = Conv2D(filters=nb_classes, 
#     #            kernel_size=(1, 1))(vgg16.output)
#     # x = Conv2DTranspose(filters=nb_classes, 
#     #                     kernel_size=(64, 64),
#     #                     strides=(32, 32),
#     #                     padding='same',
#     #                     activation='sigmoid',
#     #                     kernel_initializer=Constant(bilinear_upsample_weights(32, nb_classes)))(x)

#     # deconv, dilation_rate ??
#     # deconv1_1 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu',
#     #                             kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(vgg16.output)

#     deconv1_1 = Conv2DTranspose(filters = n_filters[0], kernel_size = (4, 4), strides = (2, 2),
#                                 padding = 'same', activation = 'relu', name = 'deconv1_1')(vgg16.output)

#     # deconv1_2 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_1)
#     # deconv1_3 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_2)
#     # deconv1_4 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_3)
    
#     # deconv1_4 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_1)

#     # deconv1_4 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu')(deconv1_1)
    
#     # deconv2_1 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv1_4)

#     # deconv2_1 = Conv2DTranspose(filters = n_filters[0], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu')(deconv1_1)

#     deconv2_1 = Conv2DTranspose(filters = n_filters[1], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu')(deconv1_1)

#     # deconv2_2 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv2_1)
#     # deconv2_3 = Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 512, 512)))(deconv2_2)
#     # deconv2_4 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 512)))(deconv2_3)
    
#     # deconv2_4 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 512)))(deconv2_1)

#     # deconv2_4 = Conv2DTranspose(filters = n_filters[1], kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu')(deconv2_1)
    
#     # deconv3_1 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 256)))(deconv2_4)

#     # deconv3_1 = Conv2DTranspose(filters = n_filters[1], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu')(deconv2_4)

#     deconv3_1 = Conv2DTranspose(filters = n_filters[2], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu')(deconv2_1)

#     # deconv3_2 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 256)))(deconv3_1)
#     # deconv3_3 = Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 256, 256)))(deconv3_2)
#     # deconv3_4 = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 128, 256)))(deconv3_3)

#     # deconv3_4 = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 128, 256)))(deconv3_1)

#     # deconv3_4 = Conv2DTranspose(filters = n_filters[2], kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu')(deconv3_1)

#     # deconv4_1 = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 128, 128)))(deconv3_4)

#     # deconv4_1 = Conv2DTranspose(filters = n_filters[2], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu')(deconv3_4)

#     deconv4_1 = Conv2DTranspose(filters = n_filters[3], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu')(deconv3_1)

#     # deconv4_2 = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 128, 128)))(deconv4_1)
#     # deconv4_3 = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 64, 128)))(deconv4_2)
    
#     # deconv4_3 = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 64, 128)))(deconv4_1)

#     # deconv4_3 = Conv2DTranspose(filters = n_filters[3], kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu')(deconv4_1)

#     # deconv5_1 = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 64, 64)))(deconv4_3)

#     # deconv5_1 = Conv2DTranspose(filters = n_filters[3], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu')(deconv4_3)

#     deconv5_1 = Conv2DTranspose(filters = n_filters[4], kernel_size = (4, 4), strides = (2, 2), padding = 'same', activation = 'relu')(deconv4_1)

#     # deconv5_2 = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'same', activation = 'relu',
#     #                                 kernel_initializer = Constant(bilinear_upsample_weights(2, 64, 64)))(deconv5_1)
#     # conv5_3 = Conv2D(filters = 1, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'sigmoid', use_bias = False)(deconv5_2)

#     conv5_3 = Conv2D(filters = n_filters[5], kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'sigmoid', use_bias = True)(deconv5_1)
#     # conv5_3 = Conv2D(filters = 1, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation = 'tanh', use_bias = False)(deconv5_1)

#     model = Model(inputs = inputs, outputs = conv5_3)

#     for layer in model.layers[:15]:
#         layer.trainable = False

#     model.summary()

#     return model

# def load_image(path):
#     img_org = Image.open(path)
#     w, h = img_org.size
#     img = img_org.resize(((w//32)*32, (h//32)*32))
#     img = np.array(img, dtype=np.float32)
#     x = np.expand_dims(img, axis=0)
#     x = preprocess_input(x)
#     return x

# def load_label(path):
#     img_org = Image.open(path)
#     w, h = img_org.size
#     img = img_org.resize(((w//32)*32, (h//32)*32))
#     img = np.array(img, dtype=np.uint8)
#     img[img==255] = 0
#     y = np.zeros((1, img.shape[0], img.shape[1], nb_classes), dtype=np.float32)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             y[0, i, j, img[i][j]] = 1
#     return y

# def generate_arrays_from_file(path, image_dir, label_dir):
#     while 1:
#         f = open(path)
#         for line in f:
#             filename = line.rstrip('\n')
#             path_image = os.path.join(image_dir, filename+'.jpg')
#             path_label = os.path.join(label_dir, filename+'.png')
#             x = load_image(path_image)
#             y = load_label(path_label)
#             yield (x, y)
#         f.close()

# # def model_predict(model, input_path, output_path):
# def model_predict(model, epoch, img, s_img, out_path):
#     # img_org = Image.open(input_path)
#     # w, h = img_org.size
#     # # img = img_org.resize(((w//32)*32, (h//32)*32))
#     # img = img_org.resize(((w // 16) * 16, (h // 16) * 16))
#     # # img = np.asarray(img, dtype = np.float32)
#     # img = np.array(img, dtype = np.float32)
#     # x = np.expand_dims(img, axis = 0)

#     # # x = preprocess_input(x)
#     # x = preprocess_input(x, mode = 'tf')

#     # with open('img.txt', 'w', encoding = 'utf-8') as f:
#     #     f.write(str(img.shape) + '\n\n')

#     #     for y in range(img.shape[1]):
#     #         s = ''
#     #         for x in range(img.shape[2]):
#     #             s += str(img[0][y][x]) + ', '
            
#     #         f.write(s[:len(s) - 2] + '\n')

#     # with open('s_img.txt', 'w', encoding = 'utf-8') as f:
#     #     f.write(str(s_img.shape) + '\n\n')

#     #     for y in range(s_img.shape[1]):
#     #         s = ''
#     #         for x in range(s_img.shape[2]):
#     #             s += str(s_img[0][y][x]) + ', '
            
#     #         f.write(s[:len(s) - 2] + '\n')

#     # pred = model.predict(img)
#     # pred = pred[0].argmax(axis=-1).astype(np.float32)

#     pred = model.predict(img)[0]
#     pred = np.squeeze(pred, axis = 2)

#     # logging.debug('pred type = %s' % type(pred))
#     # logging.debug('pred shape = %s' % str(pred.shape))
    
#     # change the range from [-1, 1] to [0, 255]
#     # pred += 1
#     # pred *= 127.5

#     # logging.info('pred shape = %s' % str(pred.shape))
#     # logging.info('pred[299][56] = %f' % pred[299][56])
#     # logging.info('pred[123][542] = %f' % pred[123][542])

#     # with open('pred_%d.txt' % epoch, 'w', encoding = 'utf-8') as f:
#     #     f.write(str(pred.shape) + '\n\n')

#     #     for y in range(pred.shape[0]):
#     #         s = ''
#     #         for x in range(pred.shape[1]):
#     #             s += str(pred[y][x]) + ', '
            
#     #         f.write(s[:len(s) - 2] + '\n')

#     pred *= 255.0

#     # with open('after_pred_%d.txt' % epoch, 'w', encoding = 'utf-8') as f:
#     #     f.write(str(pred.shape) + '\n\n')

#     #     for y in range(pred.shape[0]):
#     #         s = ''
#     #         for x in range(pred.shape[1]):
#     #             s += str(pred[y][x]) + ', '
            
#     #         f.write(s[:len(s) - 2] + '\n')

#     out_img = Image.fromarray(pred, mode='P')
#     # img = img.resize((w, h))

#     # logging.info(out_img.size)
#     # logging.info('out_img size = %s' % (out_img.size))
#     # logging.info('out_img.getpixel((56, 299)) = %d' % out_img.getpixel((56, 299)))
#     # logging.info('out_img.getpixel((542, 123)) = %d' % out_img.getpixel((542, 123)))

#     # with open('out_img_%d.txt' % epoch, 'w', encoding = 'utf-8') as f:
#     #     w, h = out_img.size
#     #     f.write('%d, %d\n\n' % (w, h))

#     #     for y in range(h):
#     #         s = ''
#     #         for x in range(w):
#     #             s += str(out_img.getpixel((x, y))) + ', '

#     #         f.write(s[:len(s) - 2] + '\n')

#     # palette_im = Image.open('palette.png')
#     # img.palette = copy.copy(palette_im.palette)
#     # img.save(output_path)
#     out_img.save(out_path)