import os
import cv2
import pydicom
import logging
from pathlib import Path
import click
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial


SEED=42
CPU=1

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('N_PATIENTS', '-n', default=0, show_default=True, type=click.IntRange(min=0, max=1775, clamp=True))
def main(input_dir, output_dir, N_PATIENTS):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.setLevel('DEBUG')
    
    if not os.path.exists(output_dir):
        logger.info(f'creating output directory: {output_dir}')
        os.makedirs(output_dir)

    imgs_fp = glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpeg')) + glob(os.path.join(input_dir, '*.dcm'))
    if not len(imgs_fp):
        logger.info(f'found 0 images in input folder ({input_dir}). quitting..')
        return
    # patient_ids = pd.Series(imgs_fp, name='fp').str.split(os.path.sep, expand=True).iloc[:, -1].str.split('_', expand=True).iloc[:, 0]
    patient_ids = [f.split(os.path.sep)[-1].split('_')[0] for f in imgs_fp]
    patient_ids = np.sort(np.unique(patient_ids))
    logger.info(f'found {len(imgs_fp)} images from {len(patient_ids)} patients')

    # patient_ids = [f for f in patient_ids if '666' in f]

    if N_PATIENTS:
        np.random.seed(SEED)
        patient_ids = np.random.choice(patient_ids, N_PATIENTS, replace=False)
    # subset
    imgs_fp = [fp for fp in imgs_fp if any([pid in fp for pid in patient_ids])]

    logger.info(f'ready to preprocess {len(imgs_fp)} images')
    logger.info(f'preprocessing parameters: outline erosion radius={outline_erosion_radius}, artifact lower color th={artifact_lower_t}, artifact min area={artifact_min_area}')

    pool = mp.Pool(CPU)

    for img_fp in tqdm(imgs_fp):
        dest_fn = img_fp.split(os.path.sep)[-1]
        dest_fn = '.'.join(dest_fn.split('.')[:-1]) + '.png'  # fixed extension png
        dest_fp = os.path.join(output_dir, dest_fn)
        if os.path.exists(dest_fp) and not overwrite_preprocessing:
            continue

        if img_fp.endswith('dcm'):
            mamm = pydicom.dcmread(img_fp).pixel_array
        else:
            mamm = cv2.imread(img_fp)

        mamm = mamm.astype(np.float32) / np.max(mamm)
        new_callback_function = partial(write_mamm, dest_fp=dest_fp)
        # mamm = preprocess_mamm(mamm, outline_erosion_radius, artifact_lower_t, artifact_left_w, artifact_min_area)
        pool.apply_async(preprocess_mamm, 
            args=(mamm, outline_erosion_radius, artifact_lower_t, artifact_left_w, artifact_min_area), 
            callback=new_callback_function)
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    
    prepr_imgs_fp = glob(os.path.join(output_dir, '*.png'))
    logger.info(f'found {len(prepr_imgs_fp)} preprocessed images')
    
        
    return

def write_mamm(mamm, dest_fp):
    cv2.imwrite(dest_fp, (mamm * 255).astype(np.uint8))
    return

def left_mamm(mamm):
    if mamm[:, :200, ...].sum() < mamm[:, -200:, ...].sum():
            mamm[:, :, ...] = mamm[:, ::-1, ...]

    return mamm

def get_act_width(mamm):
    w = mamm.shape[1] // 3

    while mamm[:, w:].max() > 0:
        w += 1

    return w

def cut_mamm(mamm, act_w, first_n_col_to_drop=0):
    h = mamm.shape[0]
        # mamm[k] = v[:h - (h % 16), :act_w + (-act_w % 16)]
    mamm = mamm[:h, first_n_col_to_drop:act_w]

    # assert mamm['mamm'].shape[0] % 16 == mamm['mamm'].shape[1] % 16 == 0

    return mamm

def clean_mamm(mamm):
    background_val = 0
    mamm[:10, :, ...] = 0
    mamm[-10:, :, ...] = 0
    mamm[:, -10:, ...] = 0
    
    if len(mamm.shape)>2:
        msk1 = (mamm[:, :, 0] == mamm[:, :, 1]) & (mamm[:, :, 1] == mamm[:, :, 2])
        mamm = mamm.mean(axis=2) * msk1
        
    msk = np.uint8((mamm > background_val) * 255)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)))

    comps = cv2.connectedComponentsWithStats(msk)
    # Find largest area, discard others
    common_label = np.argmax(comps[2][1:, cv2.CC_STAT_AREA]) + 1

    msk = (comps[1] == common_label).astype(np.uint8)

    mamm[:, :] = msk * mamm[:, :]

    return mamm

def remove_outline(mamm, erosion_size=200):
    """
    "We firstly applied morphological erosion to a structure element radius of
    100 pixels to remove the pixels close to the breast outline."
    """
    radius = erosion_size
    erosion_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(radius, radius))
    eroded_img = cv2.erode(mamm, erosion_struct)
    erosion_mask = eroded_img > 0
    # erosion_mask = (eroded_img > 10).astype(np.uint8)
    # image_erosion = mamm.copy()
    mamm *= erosion_mask/np.max(erosion_mask)
    # mamm *= (erosion_mask/np.max(erosion_mask))
    
    return mamm

def clean_border_artifacts(mamm, th=180, w=100, min_area=1000):
    """ Remove bright areas near the border
    t is the lower threshold for color intensity
    w is the max distance from left border where to cut and search for artifacts
    """
    artifact_mask = cv2.threshold(np.uint8(mamm[:, :w]*255), th, 255, cv2.THRESH_BINARY)[1]

    mask = np.zeros(mamm.shape).astype(np.uint8)
    mask[:, :w] = artifact_mask
    
    if np.sum(mask!=0) < min_area:
        return mamm

    comps = cv2.connectedComponentsWithStats(mask)

    # reset mask, add only masks that satisfy following rules
    mask = np.zeros(mamm.shape).astype(np.uint8)
    
    # Loop through all areas and filter those not eligible
    # Discard first result which is the whole image
    for common_label, area in enumerate(comps[2][1:, cv2.CC_STAT_AREA], start=1):
        if area < min_area:
            continue
        m = (comps[1] == common_label)#.astype(np.uint8)
        if np.count_nonzero((mamm*m)>=th/np.max(mamm)) < min_area:
            continue
        # Should be attached to the border -> 
        # 1/50th of all pixel should be in the first two columns
        if np.sum(m[:, :2]) < np.sum(m)/50:
            continue
        mask = np.logical_or(mask, m)#(mask + m).astype(bool)
    
    mask = mask.astype(np.uint8)

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(10, 10)))
    
    mask = 1 - mask  # invert selection, discard artifacts

    mamm[:, :] = mask * mamm[:, :]

    return mamm

def preprocess_mamm(mamm, outline_erosion_radius=200, artifact_lower_t=180, artifact_left_w=100, artifact_min_area=2000):
  # put it on the left
  mamm = left_mamm(mamm)
  # 
  mamm = clean_mamm(mamm)
  # identify column where background starts
  act_w = get_act_width(mamm)
  mamm = cut_mamm(mamm, act_w, first_n_col_to_drop=1)
  # erode to remove breast outline
  mamm = remove_outline(mamm, outline_erosion_radius)
  act_w = get_act_width(mamm)
  mamm = cut_mamm(mamm, act_w, )

  mamm = clean_border_artifacts(mamm, artifact_lower_t, artifact_left_w, artifact_min_area)

  return mamm



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    overwrite_preprocessing = False
    outline_erosion_radius = 200
    artifact_lower_t = 180
    artifact_left_w = 100
    artifact_min_area = 1000

    main()