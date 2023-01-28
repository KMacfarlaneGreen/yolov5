import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import json
from PIL import Image

def read_json_coords(label):
    '''Read a json file containing bounding boxes into coordinate arrays '''
    coords, centres = [], []
    
    for object in label['shapes']:
        # get coordinates of lower left & upper right corners of each bounding box
        [[x1, y1], [x3, y3]] = object['points'][0], object['points'][1]
        coords.append([[x1, y1], [x3, y3]])

        # store centre coordinate of each bounding box for clustering
        centres.append([(x1+x3)//2, (y1+y3)//2])
        coords_array = np.array(coords)
        centres_array = np.array(centres)
    return coords_array, centres_array

def get_bbox_info_aerial(box_path):
    '''Convert json bounding box file to correct image format'''
    with open(box_path, 'r') as f:
        label = json.load(f)
        coords, centres = read_json_coords(label)
       
        ## DB-Scan algorithm for clustering ##
    
        eps = 1 # threshold distance between two points to be in the same 'neighbourhood'
        dbscan = DBSCAN(min_samples=1, eps=eps)
        y = dbscan.fit_predict(centres)

        # storing coordinates of clusters, relative to boundaries of image (not tile)
        info = {}
        for i in range(y.max()+1):
        
            # calculate the max and min coords of all the bounding boxes in the cluster
            box_centres = centres[np.where(y==i)[0]]
            min_x, max_x = box_centres[:, 0].min(), box_centres[:, 0].max()
            min_y, max_y = box_centres[:, 1].min(), box_centres[:, 1].max()
        
            # assign each cluster of objects as an item
            item = {}
            item['centre'] = [(min_x+max_x)//2, (min_y+max_y)//2]
            item['object_boxes'] = coords[np.where(y==i)[0]].tolist()
            for name in label['shapes']:
                if name['label'] == "whale":
                    item['name'] = "whale"
                if name['label'] == "nonwhale":
                    item['name']= "nonwhale"
            info[i] = item

        return(info)

def save_aerial_files(image_path, box_path, output_dir, input_name):
    
    '''
    save each image as .png
    along with corresponding bounding box labels in YOLO format as .txt
    '''
    image = Image.open(image_path)
    info = get_bbox_info_aerial(box_path)
    
    # initialise a figure to visualise output tiles
    fig = plt.figure(figsize=(20, 100))
    n_tiles = len(info.keys())
    
    for i, k in enumerate(info.keys()):     
                             
        # get centre of each bounding box cluster
        x, y = info[k]['centre'][0], info[k]['centre'][1] # in pixels, with origin in lower (upper) left

        width, height = image.getbbox()[2], image.getbbox()[3]      #2,3 are right and lower bounds
               
        # save image 
        image_name = '{}/{}.png'.format(output_dir, input_name)

        imwrite(image_name, image)

        ax = fig.add_subplot(n_tiles, 4, k+1)
        ax.imshow(image)
                
        # save label file
        file = open(image_name.replace('.png', '.txt'), 'a')
        
        for j, box in enumerate(info[k]['object_boxes']):
            # get coordinates of upper left (x1) and lower right (x3) corner of bounding box
            [[x1, y1], [x3, y3]] = box # in px, origin in lower left

            # define bounding box centre & width
            box_centre_x = (x1+x3)//2
            box_centre_y = (y1+y3)//2
            box_width = x3-x1
            box_height = y3-y1
            
            # add bounding boxes to tile subplot
            rect = patches.Rectangle((x1, y1), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            title = str(k+1)
            ax.set_title(title)

            # write label to .txt file
            
            if info[k]['name'] == 'whale' : 
                lab = '0 {} {} {} {}\n'.format(abs(box_centre_x/width), abs(box_centre_y/height), abs(box_width/width), abs(box_height/height))
            if info[k]['name'] == 'nonwhale' :
                lab = '1 {} {} {} {}\n'.format(abs(box_centre_x/width), abs(box_centre_y/height), abs(box_width/width), abs(box_height/height))
            file.write(lab)
        file.close()
    plt.show()


