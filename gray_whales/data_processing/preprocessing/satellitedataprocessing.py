import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import json

def convert_coords(image, geo_image, label, x, y): 
    
    ''' 
    convert from geographic coordinates to image-scaled coordinates, relative to bottom left of image
    '''

    # get coordinates defining extent of input image and labels
    bbox = geo_image.bounds
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # get distance of point x,y from image origin
    x_relative = x - x_min
    y_relative = y - y_min
        
    # get scaling factors
    width = x_max - x_min
    height = y_max - y_min
    (width_pix, height_pix) = image.size

    
    x_pix = x_relative/width * width_pix
    y_pix = y_relative/height * height_pix
    
    return x_pix, y_pix

def read_coords(label):
    '''Read a file containing bounding boxes into coordinate arrays'''
    coords, centres = [], []
    
    for object in label['features']:
        # get coordinates of lower left & upper right corners of each bounding box
        [[x1, y1], [x3, y3]] = object['geometry']['coordinates'][0][0], object['geometry']['coordinates'][0][2]
        coords.append([[x1, y1], [x3, y3]])

        # store centre coordinate of each bounding box for clustering
        centres.append([(x1+x3)//2, (y1+y3)//2])
        coords_array = np.array(coords)
    return coords_array, centres

def get_bbox_info(box_path, image, geotif):
    '''Convert bounding box file to correct image format'''

    with open(box_path, 'r') as f:
        label = json.load(f)
        coords, centres = read_coords(label)
        # convert bounding box coordinates from geographic to image-scaled
        centres_converted = np.array([convert_coords(image, geotif, label, point[0], point[1]) for point in centres])
    
        coords_converted = np.array([[convert_coords(image, geotif, label, point[0][0], point[0][1]), 
                                  convert_coords(image, geotif, label, point[1][0], point[1][1])] 
                            for point in coords
                       ])
        
        ## DB-Scan algorithm for clustering ##
    
        eps = 250 # threshold distance between two points to be in the same 'neighbourhood'
        dbscan = DBSCAN(min_samples=1, eps=eps)
        y = dbscan.fit_predict(centres_converted)

        # storing coordinates of clusters, relative to boundaries of image (not tile)
        info = {}
        for i in range(y.max()+1):
        
            # calculate the max and min coords of all the bounding boxes in the cluster
            box_centres = centres_converted[np.where(y==i)[0]]
            min_x, max_x = box_centres[:, 0].min(), box_centres[:, 0].max()
            min_y, max_y = box_centres[:, 1].min(), box_centres[:, 1].max()
        
            # assign each cluster of objects as an item
            item = {}
            item['centre'] = [(min_x+max_x)//2, (min_y+max_y)//2]
            item['object_boxes'] = coords_converted[np.where(y==i)[0]].tolist()
            item['name'] = "whale"
            info[i] = item

        return(info)


def save_files(image,info, output_dir, input_name):
    
    '''
    for each cluster of bounding boxes, save a 512x512 image chip as .png
    along with bounding box labels in YOLO format as .txt
    '''
    
    # initialise a figure to visualise output tiles
    fig = plt.figure(figsize=(20, 100))
    n_tiles = len(info.keys())
    
    for i, k in enumerate(info.keys()):     
                             
        # get centre of each bounding box cluster
        x, y = info[k]['centre'][0], info[k]['centre'][1] # in pixels, with origin in lower left
        
        # define coordinates for cropping
        # set limits since tile boundaries cannot exceed image boundaries
        width, height = image.getbbox()[2], image.getbbox()[3]      #2,3 are right and lower bounds
        
        left, top, right, bottom = x-256, (height-y)-256, x+256, (height-y)+256 
               
        left_lim, top_lim = max(0,int(left)), max(0,int(top))
        right_lim, bottom_lim = min(width, int(right)), min(height, int(bottom))
               
        # crop and save image tiles
        image_name = '{}/{}_{}.png'.format(output_dir, input_name, k)
        image_tile = image.crop([left_lim, top_lim, right_lim, bottom_lim])


        imwrite(image_name, image_tile)

        ax = fig.add_subplot(n_tiles, 4, k+1)
        ax.imshow(image_tile)
                
        # save label file
        file = open(image_name.replace('.png', '.txt'), 'a')
        
        for j, box in enumerate(info[k]['object_boxes']):
            # get coordinates of lower left (x1) and upper right (x3) corner of bounding box
            [[x1, y1], [x3, y3]] = box # in px, origin in lower left
            
            # define bounding box coordinates relative to boundaries of current image tile
            x1_rel, x3_rel = x1-left_lim, x3-left_lim
            y1_rel, y3_rel = (height-y1)-top_lim, (height-y3)-top_lim
            
            # get some scaling factors to convert from image to tile coordinates
            tile_width = right_lim - left_lim
            tile_height = top_lim - bottom_lim

            # define bounding box centre & width
            box_centre_x = (x1_rel+x3_rel)//2
            box_centre_y = (y1_rel+y3_rel)//2
            box_width = x3-x1
            box_height = y1-y3
            
            # add bounding boxes to tile subplot
            rect = patches.Rectangle((x1_rel, y1_rel), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            title = str(k+1)
            ax.set_title(title)

            # write label to .txt file
            
            if info[k]['name'] == 'whale' : 
                lab = '0 {} {} {} {}\n'.format(abs(box_centre_x/tile_width), abs(box_centre_y/tile_height), abs(box_width/tile_width), abs(box_height/tile_height))
            if info[k]['name'] == 'not-whale' :
                lab = '1 {} {} {} {}\n'.format(abs(box_centre_x/tile_width), abs(box_centre_y/tile_height), abs(box_width/tile_width), abs(box_height/tile_height))
            file.write(lab)
        file.close()
    plt.show()

def get_nonwhale_bbox_info(box_path, image, geotif):
    
    with open(box_path, 'r') as f:
        label = json.load(f)
        coords, centres = read_coords(label)
        # convert bounding box coordinates from geographic to image-scaled
        centres_converted = np.array([convert_coords(image, geotif, label, point[0], point[1]) for point in centres])
    
        coords_converted = np.array([[convert_coords(image, geotif, label, point[0][0], point[0][1]), 
                                  convert_coords(image, geotif, label, point[1][0], point[1][1])] 
                            for point in coords
                       ])
            ## DB-Scan algorithm for clustering ##
    
        eps = 250 # threshold distance between two points to be in the same 'neighbourhood'
        dbscan = DBSCAN(min_samples=1, eps=eps)
        y = dbscan.fit_predict(centres_converted)

        # storing coordinates of clusters, relative to boundaries of image (not tile)
        info = {}
        for i in range(y.max()+1):
        
            # calculate the max and min coords of all the bounding boxes in the cluster
            box_centres = centres_converted[np.where(y==i)[0]]
            min_x, max_x = box_centres[:, 0].min(), box_centres[:, 0].max()
            min_y, max_y = box_centres[:, 1].min(), box_centres[:, 1].max()
        
            # assign each cluster of objects as an item
            item = {}
            item['centre'] = [(min_x+max_x)//2, (min_y+max_y)//2]
            item['object_boxes'] = coords_converted[np.where(y==i)[0]].tolist()
            item['name'] = "not-whale"
            info[i] = item

        return(info)

