import caffe
import numpy as np
from PIL import Image
def inference(prototxt, caffemodel, image):
    net = caffe.Net(prototxt,
            caffemodel,
            caffe.TEST)
    mean = np.ones([3, 32, 32], dtype=np.float)
    mean[0,:,:] = 128
    mean[1,:,:] = 128
    mean[2,:,:] = 128
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    # transformer.set_input_scale('data', 0.00390625)
    # transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    image = caffe.io.load_image(image)
    print(image.shape)
    transformed_image = transformer.preprocess('data', image)
    out = net.forward_all(data=np.asarray([transformed_image]))
    return out

prototxt_path = 'deploy.prototxt'
caffemodel_path = 'resnet-18.caffemodel'
img_path = 'cat_32.jpg'
labels_file = 'cifar10/batches.meta.txt'
prediction = inference(prototxt_path, caffemodel_path, img_path)
labels = np.loadtxt(labels_file, str, delimiter='\t')
output_prob = prediction['softmax'][0]
# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
print('probabilities and labels:')
for i in top_inds:
    print(i,output_prob[i], labels[i])

