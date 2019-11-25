
import config
from bs4 import BeautifulSoup
import json
from tfrecords import create_tfrecords
from tfrecords import create_tfrecords_with_val_split
import tensorflow as tf
import os
import cv2
import pdb


def main():
    # check if output_dir exist.
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    # from dataset dir, find images and annotations paths
    annot_list = []
    images_list = []
    for path, folders, files in os.walk(config.DATASET_DIR):
        for filename in sorted(files):
            if filename.endswith('.json'):
                annot_file = os.path.join(path, filename)
                annot_list.append(annot_file)
            if filename.endswith('.png'):
                image_file = os.path.join(path, filename)
                images_list.append(image_file)

    # create classes.pbtxt
    classes_file = os.path.join(config.OUTPUT_DIR, 'classes.pbtxt')
    f = open(classes_file, 'w')
    for (k, v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
        f.write(item)
    f.close()
    
    # make interface dictionary which will be split into train keys and test keys which
    # can be passed to create_tfrecods api
    data_dict = {}

    for image_path in sorted(images_list): 
        key = image_path
        path, image_name = os.path.split(image_path)
        image_basename, image_ext = os.path.splitext(image_name)
        image_dir = os.path.split(path)
        
        annot_dir = os.path.join(image_dir[0], 'ann')
        annot_name = image_name + '.json'
        annot_path = os.path.join(annot_dir, annot_name)

        # pdb.set_trace()
        image = cv2.imread(image_path)
        image_h, image_w, image_d = image.shape 
        if os.path.exists(annot_path):
            b = data_dict.get(key, [])
            # # load annotation file and load bounding boxes
            # contents = open(annot_path).read()
            # soup = BeautifulSoup(contents, 'lxml')
            # for obj in soup.findAll('object'):
            #     obj_names = obj.findChildren('name')
            #     for name_tag in obj_names:
            #         fname = soup.findChild('filename').contents[0]
            #         label = name_tag.contents[0]
            #         if label == 'audi-symbol':
            #             label = 'audi'
            #         if label == 'huawei-symbol':
            #             label = 'huawei'
            #         if label == 'redbull':
            #             continue
            #         if label == 'redbull-symbol':
            #             label = 'redbull'
            #         if label == 'starbucks-symbol':
            #             label = 'starbucks'
            #         if label in config.CLASSES.keys():
            #             bbox = obj.findChildren('bndbox')[0]
            #             startX = float(bbox.findChildren('xmin')[0].contents[0])
            #             startY = float(bbox.findChildren('ymin')[0].contents[0])
            #             endX = float(bbox.findChildren('xmax')[0].contents[0])
            #             endY = float(bbox.findChildren('ymax')[0].contents[0])

            #             if endX > image_w or endY > image_h:
            #                 print('[INFO] box size greater than image size..skipping: {}'.format(image_path))
            #                 continue
            #                 # pdb.set_trace()
            #             b.append((label, (startX, startY, endX, endY)))
            with open(annot_path, 'r') as f:
                json_tags = json.load(f)
                for i, tag in enumerate(json_tags['objects']):
                    # if json_tags['objects'][i]['tags'][0]['name'] == 'confidence':
                    #     confidence = json_tags['objects'][i]['tags'][0]['value']
                    classTitle = json_tags['objects'][i]['classTitle']
                    x1 = json_tags['objects'][i]['points']['exterior'][0][0]
                    y1 = json_tags['objects'][i]['points']['exterior'][0][1]
                    x2 = json_tags['objects'][i]['points']['exterior'][1][0]
                    y2 = json_tags['objects'][i]['points']['exterior'][1][1]
                    startX = min(x1, x2)
                    startY = min(y1, y2)
                    endX = max(x1, x2)
                    endY = max(y1, y2)
                    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.imshow('image', image)
                    # cv2.waitKey(0)
                    b.append((classTitle, (startX, startY, endX, endY)))
            data_dict[key] = b

    #create_tfrecords_with_val_split(data_dict, config.CLASSES, config.OUTPUT_DIR)
    output_file = os.path.join(config.OUTPUT_DIR, 'training.record')
    create_tfrecords(data_dict, config.CLASSES, output_file)

if __name__ == '__main__':
    main()

