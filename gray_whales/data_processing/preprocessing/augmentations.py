import os
import numpy as np
import cv2
import albumentations as A
from PIL import Image
from matplotlib import pyplot as plt
from albumentations import Compose, BboxParams
from imageio import imwrite

BOX_COLOR = (150, 0, 0)
TEXT_COLOR = (200, 200, 200)


def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='coco', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['category_id']))

def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=1):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    
    ## uncomment the following lines to display class label with bounding box
    #((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    #cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    #cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    
    ## uncomment the following lines to display bounding box
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
        
    plt.figure(figsize=(12, 12))
    plt.imshow(img)

def visualize_subplot(annotations, category_id_to_name):
    img = annotations['image'].copy()
    
    ## uncomment the following lines to display bounding boxes
    #for idx, bbox in enumerate(annotations['bboxes']):
        #img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    
    return(img)

def generate_augmentations(image_path, label_path, output_dir, image_id, n): # initialise a figure to visualise output tiles
    '''Generate n randomly selected augmentations per image'''
    bboxes = []
    with open(label_path, 'r') as f:
        label = f.read()
        image = np.asarray(Image.open(image_path))
        items = label.splitlines()
        imwidth, imheight = image.shape[0], image.shape[1]
        for i in items:
            obj = i.split()
            class_id = int(obj[0])
            x_centre = float(obj[1]) * imwidth
            y_centre = float(obj[2]) * imheight
            width = float(obj[3]) * imwidth
            height = float(obj[4]) * imheight
            bb = [None] * 4
            bb[0], bb[1], bb[2], bb[3] = x_centre-width/2, y_centre-height/2, width, height
            # x1, y1, width, height
            bboxes.append(bb)
        if items[0][0] == '0':
            annotations = {'image': image, 'bboxes': bboxes, 'category_id': [0]*len(bboxes)}
        if items[0][0] == '1':
            annotations = {'image': image, 'bboxes': bboxes, 'category_id': [1]*len(bboxes)}
        category_id_to_name = {0: 'whale', 1: 'nonwhale'}
    
    fig = plt.figure(figsize=(20, 100))

    for i in range(n):
        ax = fig.add_subplot(n, 4, i+1)
    
        aug = get_aug([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-180,180), always_apply=True),
        #A.Blur(p=0.5),
        #A.RandomBrightness(p=0.5),
        #A.RandomContrast(p=0.5),
        #A.GaussianBlur(p=0.5),
        #A.GaussNoise(p=0.5)
        ])

        augmented = aug(**annotations)
        vis = visualize_subplot(augmented, category_id_to_name)
        ax.imshow(vis)
    
        # save augmented image file
        image_name = '{}/{}_{}_{}.png'.format(output_dir, image_id, "aug", i)
        imwrite(image_name, vis)


        tile_width = vis.shape[0]
        tile_height = vis.shape[1]
    
        # save labels corresponding to augmented image
        file = open(image_name.replace('.png', '.txt'), 'a')
        for bbox in augmented['bboxes']:
        
            x1, y1, box_width, box_height = bbox[0], bbox[1], bbox[2], bbox[3]
            box_centre_x = x1+box_width/2
            box_centre_y = y1+box_height/2

            if augmented['category_id'][0] == 0:   
                lab = '0 {} {} {} {}\n'.format(abs(box_centre_x/tile_width), 
                abs(box_centre_y/tile_height), 
                abs(box_width/tile_width), 
                abs(box_height/tile_height))
            
            if augmented['category_id'][0] == 1:
                lab = '1 {} {} {} {}\n'.format(abs(box_centre_x/tile_width), 
                abs(box_centre_y/tile_height), 
                abs(box_width/tile_width), 
                abs(box_height/tile_height))
        
            file.write(lab)
        
        file.close()
    plt.show()