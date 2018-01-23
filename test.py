from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import os
import numpy as np
from PIL import Image

# datagen = ImageDataGenerator()

# t_generator = datagen.flow_from_directory(
#                 'data1/video/104_conv_L/frame',
#                 batch_size = 2)

# i = 0

batch_size = 64
data_dir = 'data'
video_dir = os.path.join(data_dir, 'video', 'test')
ori_dir = 'frame'
s_dir = 'saliency'


def load_image(path):
    if os.path.isfile(path):
        img_org = Image.open(path)
        w, h = img_org.size

        # ensure that the size of img is divisible by 16
        img = img_org.resize(((w // 16) * 16, (h // 16) * 16))
        # img = np.asarray(img, dtype = np.float32)
        x = np.array(img, dtype = np.float32)

        # x = np.expand_dims(img, axis=0)

        if len(x.shape) != 3:
            x = np.expand_dims(x, axis = 2)
            # x /= 255.0

        # in tf mode, it will scale pixels between -1 and 1 sample-wise
        x = preprocess_input(x, mode = 'tf')

        return x
    else:
        print('load_image error: path is not a file')
        exit()

def generate_arrays_from_file(image_dir, label_dir, batch_size = batch_size, is_infinite = True):
    while 1:
        for name in os.listdir(video_dir):
            video_path = os.path.join(video_dir, name)
            if os.path.isdir(video_path):
                ori_path = os.path.join(video_path, ori_dir)
                s_path = os.path.join(video_path, s_dir)

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

                    print('load video %s img %s' % (name, img_name))

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

_dir = 'data/video/test/7_7_full(320x180_0-382)_conv/saliency'

for path in os.listdir(_dir):
    print(os.path.join(_dir, path))


# while i <= t_generator.batch_index:
#     data = t_generator.next()
#     print(type(data))
#     # print(data.shape)
#     print(len(data))
#     print(type(data[0]))
#     print(data[0].shape)
#     print(data[1].shape)
#     print(data[1])
#     exit()

# data = generate_arrays_from_file.next()
# print(data)

# generator = generate_arrays_from_file(ori_dir, s_dir, is_infinite = False)

# # print(len(generator))

# i = 0

# for data in generator:
#     # print()
#     i += 1

# print(i)


# for data in generator:
#     print(type(data))
#     print(type(data[0]))
#     print(data[0].shape)
#     print(type(data[1]))
#     print(data[1].shape)


# arr = np.array([[254.5, 1.2], [123.4, 212.8], [55.5, 0.5]])

l = [[254.5, 1.2], [123.4, 212.8]]
ll = []
ll.append(l)

print(l)
print(ll)
del l
print(ll)

# l_arr = np.array(l)
# l_asarr = np.asarray(l_arr)

# print(l)
# print(l_arr)
# print(l_asarr)

# del l
# print(l_arr)
# print(l_asarr)

# del l_arr

# print(l_asarr)

# arr = np.asarray([[254.5, 1.2], [123.4, 212.8]])
# print(arr)
# print(arr.shape)
# print(arr.dtype)

# arr = arr.astype(np.uint8)

# print(arr)
# print(arr.shape)

# p_img = Image.fromarray(arr, mode = 'P')
# p_img.save('p_img.png')

# l_img = Image.fromarray(arr, mode = 'L')
# l_img.save('l_img.png')

# print('p_img')
# p_img = Image.open('p_img.png', mode = 'r')

# w, h = p_img.size

# for y in range(h):
#     s = ''
#     for x in range(w):
#         s += str(p_img.getpixel((x, y))) + ', '
    
#     print(s[:len(s) - 2] + '\n')

# print('l_img')
# l_img = Image.open('l_img.png', mode = 'r')

# w, h = l_img.size

# for y in range(h):
#     s = ''
#     for x in range(w):
#         s += str(l_img.getpixel((x, y))) + ', '
    
#     print(s[:len(s) - 2] + '\n')

