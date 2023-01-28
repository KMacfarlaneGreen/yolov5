import numpy as np
import rasterio

def rgb_array(image_path):
    '''image path to rgb image array'''
    rasterio_image = rasterio.open(image_path)
    red = rasterio_image.read(1)
    green = rasterio_image.read(2)
    blue = rasterio_image.read(3)

    rgb_image_array =np.dstack((blue,green,red)) 

    return rgb_image_array

def max_norm(image_path):
    '''Normalize image scaled using max value'''
    rasterio_image = rasterio.open(image_path)
    red = rasterio_image.read(1)
    green = rasterio_image.read(2)
    blue = rasterio_image.read(3)

    #scales the values to between 0 and 255
    norm_red = (red)*(255/red.max())
    norm_green = green*(255/green.max())
    norm_blue = blue*(255/blue.max())

    rgb_max_norm = np.dstack((norm_blue, norm_green, norm_red))

    return rgb_max_norm 

def perc_norm(image_path):
    '''Normalize image scaled using 95th percentile'''
    rasterio_image = rasterio.open(image_path)
    red = rasterio_image.read(1)
    green = rasterio_image.read(2)
    blue = rasterio_image.read(3)

    #scales the values to between 0 and 255
    norm_red = (red)*(255/np.percentile(red,98))
    norm_green = green*(255/np.percentile(green,98))
    norm_blue = blue*(255/np.percentile(blue,98))

    rgb_perc_norm = np.dstack((norm_blue, norm_green, norm_red))

    return rgb_perc_norm   
