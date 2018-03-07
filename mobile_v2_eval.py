import mxnet as mx
from images_dataset import images_to_iter
import numpy as np

def load_model(prefix, epoch=0, batch_size=1, data_shape=(3, 512, 512)):
    symbol, args, auxs = mx.model.load_checkpoint(prefix, epoch)
    mod = mx.mod.Module(symbol, label_names=('prob_label',), context=mx.gpu(0))
    data_shape = data_shape
    mod.bind(data_shapes=[('data', (batch_size, data_shape[0], data_shape[1], data_shape[2]))],
                  label_shapes=[('prob_label', (batch_size,1000,1))],
                  grad_req='null')
    mod.set_params(args, auxs)
    return mod

def process_image(image_lists):
    outputs = []
    for image in image_lists:
        nh, nw = 224, 224
        h, w, _ = image.shape
        if h < w:
            off = (w - h) / 2
            image = image[:, off:off + h]
        else:
            off = (h - w) / 2
            image = image[off:off + h, :]
        outputs.append(image)
    return outputs


if __name__ == "__main__":
    data_iter = images_to_iter(['cat.jpg',],(3,224,224),(103.94, 116.78, 123.68),1,process_image)
    model = load_model("mxnet_model/mobilev2", epoch=0, batch_size=1, data_shape=(3, 224, 224))

    result = model.predict(data_iter)
    result = result.asnumpy()
    result = np.squeeze(result)
    idx = np.argsort(-result)
    label_names = np.loadtxt('synset.txt', str, delimiter='\t')
    for i in range(5):
        label = idx[i]
        print (result[label], label_names[label])



