import caffe
import numpy as np
import cv2 as cv
def inference(prototxt, caffemodel, image):
    net = caffe.Net(prototxt,
            caffemodel,
            caffe.TEST)
    image = cv.imread(image)
    print(image.shape)
    image = image.transpose((2,0,1))
    print(image)
    out = net.forward_all(data=np.asarray([image]))
    return out

prototxt_path = 'deploy.prototxt'
caffemodel_path = 'resnet-18.caffemodel'
img_path = '0_tench.jpg'
labels_file = 'synset_words.txt'
prediction = inference(prototxt_path, caffemodel_path, 'transformed/' + img_path)
labels = np.loadtxt(labels_file, str, delimiter='\t')
output_prob = prediction['prob'][0]
# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
print('probabilities and labels:')
for i in top_inds:
    print(i,output_prob[i], labels[i])

