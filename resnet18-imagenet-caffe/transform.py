import caffe
import numpy as np
from PIL import Image
import cv2 as cv
def inference(prototxt, caffemodel, image):
    net = caffe.Net(prototxt,
            caffemodel,
            caffe.TEST)
    mean = np.ones([3, 224, 224], dtype=np.float)
    mean[0,:,:] = 104
    mean[1,:,:] = 117
    mean[2,:,:] = 123
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    image = caffe.io.load_image(image)
    transformed_image = transformer.preprocess('data', image)
    print(transformed_image)
    out = net.forward_all(data=np.asarray([transformed_image]))
    return transformed_image,out

prototxt_path = 'deploy.prototxt'
caffemodel_path = 'resnet-18.caffemodel'
img_path = 'printer.jpg'
labels_file = 'synset_words.txt'
transformed_image, prediction = inference(prototxt_path, caffemodel_path, 'raw/'+img_path)
labels = np.loadtxt(labels_file, str, delimiter='\t')
output_prob = prediction['prob'][0]
# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
print('probabilities and labels:')
for i in top_inds:
    print(i,output_prob[i], labels[i])

dirname = 'transformed/'
filepath = dirname + str(int(top_inds[0])) + '_' + img_path
print(filepath)
saveim = transformed_image.transpose((1, 2, 0))
cv.imwrite(filepath, saveim)
# saveim = Image.fromarray(saveim.astype(np.uint8))
# print('-----------------')
# print(np.array(saveim))
# saveim.save(filepath)
# loadim = Image.open(filepath)
# print('+++++++++++++++')
# print(np.array(loadim))
