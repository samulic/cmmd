import cv2
import numpy as np
import png
import os
import pydicom


def get_png_filename(row):
    """ 'D1-0000_L_A-R_mass_malignant|triple negative.png'
    """
    fname = get_filename(row)
    fname += ".png" # if not row['PatientName'] in PIDS_TO_CHECK else '-wrong.png'
    return fname
    
def get_filename(row):
    fname = row['PatientName']
    fname += f"_{row['ImageLaterality']}"
    fname += f"_{'-'.join(eval(row['PatientOrientation']))}"
    fname += f"_{row['abnormality']}"
    fname += f"_{row['classification']}{'|'+row['subtype'] if row['subtype'] else ''}"
    return fname

def erode(image, ksize=(10,10)):
    return cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize))

def tophat(image, ksize=(10,100)):
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize))

def otsu(image):
    threshold, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return mask

# def quantile_threshold(image, quantile):
#     lb = np.quantile(image, quantile)
#     threshold, mask = cv2.threshold(image, lb, 255, cv2.THRESH_BINARY)
#     return mask, threshold

# def fixed_threshold(image, t):
#     lb = t
#     threshold, mask = cv2.threshold(image, lb, 255, cv2.THRESH_BINARY)
#     return mask, threshold

def get_threshold_mask(image, t):
    if isinstance(t, str):
        if t.lower() != 'otsu':
            print('Unknown method named ', t)
            return None, None
        threshold, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return mask, threshold
    lb = np.quantile(image, t) if t < 1 else t
    threshold, mask = cv2.threshold(image, lb, 255, cv2.THRESH_BINARY)
    return mask, threshold

def dilate(mask, ksize=(10,10)):
    return cv2.dilate(mask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize))

def bbox(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(str(len(cnts))+' contours detected')

    # Find maximum area contour
    area = np.array([cv2.contourArea(cnts[i]) for i in range(len(cnts))])
    if np.sum(area)>1:
        maxa_ind = np.argmax(area) # index of maximum area contour
        xx = [cnts[maxa_ind][i][0][0] for i in range(len(cnts[maxa_ind]))]
        yy = [cnts[maxa_ind][i][0][1] for i in range(len(cnts[maxa_ind]))]
        return [min(xx),max(xx),min(yy),max(yy)]
    return None

def identify_bbox(image, erosion_ksize=(150,150), tophat_ksize=(160, 20), dilatation_ksize=(130,130)):
  """ Automatically identify potential lesions as in 
  https://downloads.hindawi.com/journals/cmmm/2019/2717454.pdf Sec 3.2.1. ROIC Extraction
  Image is an openCV array of gray colors or an image filepath
  """
  if isinstance(image, str):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = erode(image, erosion_ksize)
  image = tophat(image, tophat_ksize)
  image = otsu(image)
  image = dilate(image, dilatation_ksize)
  roi = bbox(image)
  return roi

def identify_bbox_v2(image, threshold=.999, erosion_ksize=(150,150), tophat_ksize=(160, 20), dilatation_ksize=(130,130)):
    """ Automatically identify potential lesions as in 
    https://downloads.hindawi.com/journals/cmmm/2019/2717454.pdf Sec 3.2.1. ROIC Extraction
    Image is an openCV array of gray colors or an image filepath
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    eroded_img = erode(image, erosion_ksize)
    erosion_mask = eroded_img > 10
    image_erosion = image.copy()
    image_erosion *= erosion_mask

    image = tophat(image_erosion, tophat_ksize)

    mask, th = get_threshold_mask(image, threshold)

    mask = dilate(mask, dilatation_ksize)

    roi = bbox(mask)

    return roi



def dicom_to_png(file, dest_fp, overwrite=False, greyscale=True):
    if os.path.exists(dest_fp) and not overwrite:
      return
    ds = pydicom.dcmread(file)

    image_2d = ds.pixel_array#.astype(float)
    shape = image_2d.shape
    if shape != (2294, 1914):
    #   logger.warning(f'Shape for image {file} is {shape} instead of (2294, 1914)')
      print(f'Shape for image {file} is {shape} instead of (2294, 1914)')

    # Write the PNG file
    with open(dest_fp, 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=greyscale)
        w.write(png_file, image_2d)#_scaled)
    return
