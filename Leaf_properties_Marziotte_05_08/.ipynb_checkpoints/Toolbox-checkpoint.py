import numpy as np
import pandas as pd

from skimage.morphology import area_opening, disk, binary_closing
from skimage.measure import find_contours, label, regionprops, regionprops_table

def leaves_props(RGB, opening=2000,closing=15, height=125, width=80):
    """ 
    This function uses a picture containing leaves as an input. 
    Returns leaves contours and a table with leaves properties.
    
    Inputs: RGB, opening, closing, height, width.
       
       RGB: Picture to be analized.
       
       opening: Will be used in area_opening (check skimage.morphology). Default is 2000.
       
       closing: Will be used in binary_closing (check skimage.morphology). Default is 15.
       
       height: Height of the picture, will be used to calculate the size of the pixel. In centimiters. Default is 125 cm.
       
       width: Width of the picture, will be used to calculate the size of the pixel. In centimiters. Default is 80 cm.
    
    Outputs: contours, properties
    
        contours: Is the contour of each leaf inside the picture.
        
        properties: Table of properties including area, axis_minor_length (length of the minor axis), axis_major_length (length of the major axis), and image.
               The first three variables are given in centimiters. Image is the image of each leaf.
               
               
    
    """
  
    #Compute factor for converting pixels into cm
    area = height*width 
    pixel_area = area/RGB.size
    
    #Define BW, and what is green
    BW = 0
    red = RGB[:, :, 0]
    green = RGB[:, :, 1]
    blue = RGB[:, :, 2]
    red_green_ratio = red/(green+1e-10)
    blue_green_ratio = blue/(green+1e-10)
    ExG = 2*green - red - blue
    
    #Saving black and white
    BW = np.logical_and(red_green_ratio<0.99, blue_green_ratio<0.99, ExG >20)
    BW = area_opening(BW, opening)
    BW = binary_closing(BW, disk(closing))
    
    #label leaves, contur and props.
    label_image = label(BW)
    contours = find_contours(BW, 0)

    properties = regionprops_table(label_image, properties = ('area',
                                                   'axis_minor_length',
                                                   'axis_major_length',
                                                   'image'))
    properties['area'] = properties['area']*pixel_area
    properties['axis_minor_length'] = properties['axis_minor_length']* np.sqrt(pixel_area)
    properties['axis_major_length'] = properties['axis_major_length']* np.sqrt(pixel_area)
    
    
    return contours, properties