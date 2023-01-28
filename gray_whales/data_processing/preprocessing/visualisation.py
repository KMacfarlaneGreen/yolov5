import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from matplotlib.pyplot import imshow
import matplotlib.patches as patches
from PIL import Image
import rasterio
import tifffile as tiff
from src.preprocessing.satellitedataprocessing import convert_coords, save_files, read_coords

def plot_rgb_hist(image_array, lim):# tuple to select colors of each channel line
    '''Plot histogram of rgb images'''
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)

    # create the histogram plot, with three lines, one for
    # each color
    plt.xlim([0, lim])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
        image_array[:, :, channel_id], bins=lim, range=(0, lim)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")
    plt.show()

def plot_rgb_boxes(file_path, box_path, array_rgb):
    '''Plot rgb image and bounding boxes'''
    with open(box_path, 'r') as f:
        label = json.load(f)
        geotif_rgb = rasterio.open(file_path)
        PIL_image = Image.fromarray(array_rgb)
        image = PIL_image
        # create figure
        fig1 = plt.figure(figsize=(16, 8))
        ax1 = fig1.add_subplot(111, aspect='equal')
        plt.imshow(np.flipud(image), origin='lower') #  flip because imshow defines upper left as origin

        # plot bounding boxes
        for object in label['features']:
        
            # get origin and dimensions of each bounding box
            bottom_left_unconv = object['geometry']['coordinates'][0][0]
            top_right_unconv = object['geometry']['coordinates'][0][2]

            bottom_left = convert_coords(image, geotif_rgb, label, bottom_left_unconv[0], bottom_left_unconv[1])
            top_right = convert_coords(image, geotif_rgb, label, top_right_unconv[0], top_right_unconv[1])

            width = top_right[0] - bottom_left[0]
            height = top_right[1] - bottom_left[1]

            # add bounding box to figure
            rect = patches.Rectangle(bottom_left, width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
        #plt.xlim([12000,22000])
        #plt.ylim([12000,22000])
        #plt.xticks([]),plt.yticks([])
        #image_format_svg = 'svg' # e.g .png, .svg, etc.
        #image_name_svg = 'liebre_report_plot.svg'
        #image_format_png = 'png' # e.g .png, .svg, etc.
        #image_name_png = 'liebre_report_plot.png'

        #fig1.savefig(image_name_svg, format=image_format_svg, dpi=200)
        #fig1.savefig(image_name_png, format=image_format_png, dpi=200)
        plt.show

