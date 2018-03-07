import cv2
import numpy as np
import mxnet as mx


def _read_imgs(img_lists):
    imgs = []
    for img_path in img_lists:
        with open(img_path, 'rb') as fp:
            img_content = fp.read()
            img = mx.img.imdecode(img_content)
        imgs.append(img)
    return imgs


def _process_imgs(img_lists, data_shape, mean_pixels, batch_size):
    imgs = []
    if isinstance(mean_pixels,tuple):
        mean_pixels = mx.nd.array(mean_pixels).reshape((3,1,1))

    for data in img_lists:
        if type(data) is np.ndarray:
            data = mx.nd.array(data)
        data = mx.img.imresize(data, data_shape[2], data_shape[1], cv2.INTER_LINEAR)
        if data.shape[2] == 3:
            data = mx.nd.transpose(data, (2, 0, 1))
        data = data.astype('float32')
        data = data - mean_pixels
        imgs.append(data.reshape((1, data_shape[0], data_shape[1], data_shape[2])))
    data = mx.nd.concat(*imgs, dim=0)
    data = data*0.017
    test_iter = mx.io.NDArrayIter(data, batch_size= batch_size, shuffle=False)
    return test_iter


def images_to_iter(name_lists, data_shape, mean_pixels, batch_size,func=None):
    image_lists = _read_imgs(name_lists)
    if func is not None:
        image_lists = func(image_lists)
    data_iters = _process_imgs(image_lists, data_shape, mean_pixels, batch_size)
    return data_iters
