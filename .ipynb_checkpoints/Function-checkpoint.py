import numpy as np
import pandas as pd

from skimage.morphology import area_opening, disk, binary_closing
from skimage.measure import find_contours, label, regionprops, regionprops_table

def crop_measure_picture(pic_list, opening=2000,closing=15, size = 125):
    """ 
    This function uses the picture or list of pictures and returns the contours, number of leaves, a table with                 properties in pixels, and the table in centimiters
    All images must havse the same lenght in cm. 
    pic_list is the picture or list of pictures
    opening: will be used in area_opening. Default is 2000
    closing: used in binary_closing. Default is 15
    size is the length of the image. Default is 125
    
    """
    data = []
    cont = []
    #Compute factor for converting pixels into cm
    pixel_cm = pic_list[1].shape[0]/size
#Is better to have the for in the finction or outside??

    for RGB in pic_list:
        BW = 0
        red = RGB[:, :, 0]
        green = RGB[:, :, 1]
        blue = RGB[:, :, 2]
        red_green_ratio = red/(green+1e-10)
        blue_green_ratio = blue/(green+1e-10)
        ExG = 2*green - red - blue
        #Saving each black and white
        BW = np.logical_and(red_green_ratio<0.99, blue_green_ratio<0.99)
        #Define open and close
        BW = area_opening(BW, opening)
        BW = binary_closing(BW, disk(closing))
        label_image = label(BW)
        contours = find_contours(BW, 0)
        cont.append(contours)
        print(f"Image contains {len(contours)} leaves")
        props = regionprops_table(label_image, properties = ('area',
                                                       'axis_minor_length',
                                                       'axis_major_length',
                                                       'image'))
        data.append(props)
        
    df_prop = pd.DataFrame(data)
    df_cont = pd.DataFrame(cont) 
    df = df_prop.div({'area': pixel_cm, 'axis_minor_length': pixel_cm, 'axis_major_length': pixel_cm, 'image': 1})

    return cont, props, df_prop, df_cont, df