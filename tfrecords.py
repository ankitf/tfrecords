
# this file can be used to create tensorflow records for any dataset

from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature
from object_detection.utils.dataset_util import bytes_feature

import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import pdb



# A higher level api to create tf records
# data_dict = {image_path_1: [(label_0, (startX, startY, endX, endY)),
#                             (label_1, (startX, startY, endX, endY))],
#              image_path_2: [(label_1, (startX, startY, endX, endY)),
#                             (label_3, (startX, startY, endX, endY)),
#                             (label_0, (startX, startY, endX, endY))]}
def create_tfrecords_with_val_split(data_dict, classes_dict, output_dir, val_split = 0.20):
    # create training and validation splits from  data dictionary
    (trainKeys, valKeys)  = train_test_split(list(data_dict.keys()),
                                             test_size=val_split,
                                             random_state=42)
    
    validation_records = os.path.join(output_dir, 'validation.record')
    training_records = os.path.join(output_dir, 'training.record')
    datasets = [('train', trainKeys, training_records),
                ('val', valKeys, validation_records)]

    train_dict = {}
    val_dict = {}
    for key in  trainKeys:
        train_dict.update({key: data_dict[key]})
    for key in valKeys:
        val_dict.update({key: data_dict[key]})

    print('processing training samples...')
    create_tfrecords(train_dict, classes_dict, training_records)

    print('processing validation samples...')
    create_tfrecords(val_dict, classes_dict, validation_records)
    
    return 


def create_tfrecords(data_dict, classes_dict, output_filename): 
    writer = tf.python_io.TFRecordWriter(output_filename)
    total = 0
    for k in data_dict.keys():       
        # load the input image from the disk as a Tensorflow object
        encoded = tf.gfile.GFile(k, 'rb').read()
        encoded = bytes(encoded)
        # load the image from disk again, this time as a PIL object
        pilImage = Image.open(k)
        (w, h) = pilImage.size[:2]
        # parse the filename and encoding from the input path
        # filename = k.split(os.path.sep)[-1]
        # encoding = filename[filename.rfind(".") + 1:]

        basename = k.split(os.path.sep)[-1]
        encoding = basename[basename.rfind(".") + 1:]

        # initialize the annotation object used to store information
        # regarding the bounding box + labels
        tfAnnot = TFAnnotation()
        tfAnnot.image = encoded
        tfAnnot.encoding = encoding
        # tfAnnot.filename = filename
        tfAnnot.filename = k
        tfAnnot.width = w
        tfAnnot.height = h

        for (label, (startX, startY, endX, endY)) in data_dict[k]:
            # Tensorflow assumes all bounding boxes are in the range[0, 1]
            # so we need to scale them
            xMin = startX / w
            xMax = endX / w
            yMin = startY / h
            yMax = endY / h

            # uncomment this block to visulize annotations
            # import cv2
            # image = cv2.imread(k)
            # startXint = int(xMin * w)
            # startYint = int(yMin * h)
            # endXint = int(xMax * w)
            # endYint = int(yMax * h)

            # cv2.rectangle(image, (startXint, startYint),
            #               (endXint, endYint), (0, 0, 255), 2)
            # text = label
            # cv2.putText(image, text, (200, 200),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 2)
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)
            # pdb.set_trace()

            # update the bounding boxs + labels
            tfAnnot.xMins.append(xMin)
            tfAnnot.xMaxs.append(xMax)
            tfAnnot.yMins.append(yMin)
            tfAnnot.yMaxs.append(yMax)
            tfAnnot.textLabels.append(label.encode('utf8'))
            tfAnnot.classes.append(classes_dict[label])
            tfAnnot.difficult.append(0)

        # increament the total number of examples
        total += 1

        # encode the data point attributes using the Tensorflow
        # helper fns
        features = tf.train.Features(feature=tfAnnot.build())
        example = tf.train.Example(features=features)

        # add the example to the writer
        writer.write(example.SerializeToString())

    # close the writer and print diagnostic information to the user
    writer.close()
    print('samples: {}'.format(total))

    return 




# A lower layer class for tensorflow type conversiion.
class TFAnnotation:
    def __init__(self):
        # initialise the bounding box + label list
        self.xMins = []
        self.xMaxs = []
        self.yMins = []
        self.yMaxs = []
        self.textLabels = []
        self.classes = []
        self.difficult = []

        # initialise additional variables, including the image itself,
        # spatial dimensions, encoding, and filename
        self.image = None
        self.width = None
        self.height = None
        self.encoding = None
        self.filename = None

    def build(self):
        # encode the attributes using their respective TensorFlow encoding fn
        w = int64_feature(self.width)
        h = int64_feature(self.height)
        filename = bytes_feature(self.filename.encode('utf8'))
        encoding = bytes_feature(self.encoding.encode('utf8'))
        image = bytes_feature(self.image)
        xMins = float_list_feature(self.xMins)
        xMaxs = float_list_feature(self.xMaxs)
        yMins = float_list_feature(self.yMins)
        yMaxs = float_list_feature(self.yMaxs)
        textLabels = bytes_list_feature(self.textLabels)
        classes = int64_list_feature(self.classes)
        difficult = int64_list_feature(self.difficult)
        # construct the TensorFlow-compatible data dictionary
        data = {
            'image/height': h,
            'image/width': w,
            'image/filename': filename,
            'image/source_id': filename,
            'image/encoded': image,
            'image/format': encoding,
            'image/object/bbox/xmin': xMins,
            'image/object/bbox/xmax': xMaxs,
            'image/object/bbox/ymin': yMins,
            'image/object/bbox/ymax': yMaxs,
            'image/object/class/text': textLabels,
            'image/object/class/label': classes,
            'image/object/difficult': difficult,
        }
        
        return data    





