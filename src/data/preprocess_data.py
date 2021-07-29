# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
import pandas as pd
import numpy as np
import cv2
import pydicom
from tqdm import tqdm

from src.models.segmentation_model import get_model, predict

# N_PATIENTS=00  # if zero download all patients
SEED=42  # to download the same N_patients

@click.command()
@click.argument('collection_reference_fp')  #type=click.Path(exists=False))
@click.argument('output_dir', type=click.Path(exists=False))
@click.option('N_PATIENTS', '-n', default=0, show_default=True, type=click.IntRange(min=0, max=1775, clamp=True))
def main(collection_reference_fp, output_dir, N_PATIENTS):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(collection_reference_fp):
        logger.error(f'input file {collection_reference_fp} not found')
        return
    collection_metadata_fp = os.path.join(os.path.sep.join(collection_reference_fp.split(os.path.sep)[:-1]), 'dicom_attributes.csv')
    # collection_reference_fp is located in the input_data_dir
    # this info is needed to be agnostic to the folder structure
    input_dir = os.path.sep.join(collection_reference_fp.split(os.path.sep)[:-1])
    preprocessed_reference_fp = os.path.join(output_dir, collection_reference_fp.split(os.path.sep)[-1])
    if os.path.exists(preprocessed_reference_fp):
        logger.info(f'output file {preprocessed_reference_fp} already exists, skipping preprocessing')
        return preprocessed_reference_fp

    collection_ref_df = pd.read_csv(collection_reference_fp)
    logger.info(f'found {len(collection_ref_df)} file paths from {collection_reference_fp}')
    if N_PATIENTS:
        np.random.seed(SEED)
        # first column is the identifier for the scan/patient
        id_col = collection_ref_df.columns[0]
        selected_ids = np.random.choice(collection_ref_df[id_col].unique(), N_PATIENTS, replace=False)
        collection_ref_df = collection_ref_df[collection_ref_df[id_col].isin(selected_ids)]
        logger.info(f'using {len(collection_ref_df)} files from {N_PATIENTS} patients')

    logger.debug('sampling 3 collection references')
    logger.debug(f'{collection_ref_df.sample(3, random_state=SEED, replace=False)}')

    # segment images

    # Preprocess images for segmentation
    logger.info(f'preparing folders for prepocessing')
    preprocessed_ref_df = map_preprocessing_filepaths(collection_ref_df['filepath'], input_dir, output_dir)


    logger.info(f'preprocessing images, saving to {output_dir}')
    #logger.info(f'preserving original input folder structure (e.g. {os.path.join(collection_ref_df["filepath"].values[0].split(os.path.sep)[:-1]).replace(input_dir, "")}')
    identify_calcifications(preprocessed_ref_df, overwrite=False)
    logger.info(f'saving preprocessed filepath references to {preprocessed_reference_fp}')
    preprocessed_ref_df.to_csv(preprocessed_reference_fp)

    # labels_df = pd.read_excel(CLINICAL_DATA_URL, keep_default_na=False)
    # logger.info(f'clinical data has {labels_df.shape} rows,columns with {labels_df.groupby(["ID1"]).size().shape[0]} patients')

    # # CSV of dicom attributes extracted from collection
    # dicom_attributes_fn = 'dicom_attributes.csv' if not N_PATIENTS else f'dicom_attributes-{N_PATIENTS}_seed-{SEED}.csv'
    # dicom_attributes_fp = os.path.join(project_dir, 'data', 'raw', dicom_attributes_fn)

    return

def map_preprocessing_filepaths(filepaths, input_dir, output_root_dir):
    """ create destination directory structure where will save PNG files 
    return pd.DataFrame with index `original_fp` and columns `preprocessed_fp`
    """
    map_fn = dict()
    for fp in filepaths:
        # replace input extension with PNG
        output_fp = '.'.join(fp.split('.')[:-1]) + '.png'  # can handle multiple extensions for now
        output_fp = output_fp.replace(input_dir, output_root_dir)
        # create (if needed) any intermediate folder to save png
        # TODO: check folders permissions 
        os.makedirs(os.path.sep.join(output_fp.split(os.path.sep)[:-1]), exist_ok=True)
        map_fn[fp] = output_fp

    preprocessed_ref_df = pd.DataFrame(pd.Series(map_fn, name='preprocessed_fp'))
    preprocessed_ref_df.index.name = 'original_fp'
    preprocessed_ref_df = preprocessed_ref_df['preprocessed_fp']
    return preprocessed_ref_df

def identify_calcifications(maps, calcification_min_diam_mm = 0.2, suffix = '__cmask_raw', overwrite=False):
    """ preprocess images for segmentation and discard non calcifications
    """
    rerun_preprocessing = False
    outline_erosion_diam_mm = 10
    bcg_th_for_bb_removal = 0.3

    model = get_model(tag='calc_detect')

    for input_fp, output_png in tqdm(maps.items()):
        raw_pred_fp = output_png.replace('.png', f'{suffix}.png')
        pixel_spacing, ps_y = pydicom.dcmread(input_fp, stop_before_pixels=True).ImagerPixelSpacing
        calcification_min_diam = round(calcification_min_diam_mm/pixel_spacing)

        mamm = preprocess_for_segmentation(input_fp, output_png, outline_erosion_diam_mm, rerun_preprocessing)

        if os.path.exists(raw_pred_fp) and not overwrite:
            pred_prob = cv2.imread(raw_pred_fp, cv2.IMREAD_UNCHANGED)/255
        else:
            pred_prob = predict(model, mamm)
        pred_mask = np.uint8(pred_prob > 0.5)

        mask = discard_non_calcifications(mamm, pred_mask, calcification_min_diam, bcg_th_for_bb_removal)
        pred_prob[~mask] = 0

        cv2.imwrite(raw_pred_fp, np.uint8(pred_prob*255))

        # mask = discard_small_calcifications(pred_mask, calcification_min_diam)
        # 
        # mask = discard_outline_calcifications(mamm, mask, bcg_th=bcg_th_for_bb_removal)
        # cv2.imwrite(raw_pred_fp.replace(f'{suffix}.png', f'__no_outline_calc{suffix}.png'), mask)

    return

def preprocess_for_segmentation(input_fp, output_png, outline_erosion_diam_mm=10, overwrite=False):
    if not overwrite and os.path.exists(output_png):
        return cv2.imread(output_png, cv2.IMREAD_UNCHANGED)

    dicom = pydicom.dcmread(input_fp)
    pixel_spacing, ps_y = map(float, dicom.ImagerPixelSpacing)
    pixel_spacing = float(pixel_spacing)
    outline_erosion_diam = round(outline_erosion_diam_mm/pixel_spacing)

    mamm = dicom.pixel_array
    mamm = preprocess_mamm(mamm, outline_erosion_diam)
    cv2.imwrite(output_png, mamm)
    
    return cv2.imread(output_png, cv2.IMREAD_UNCHANGED)

def discard_non_calcifications(image, mask, min_diam, bcg_th=.3):
    """ 
        - remove blobs with greatest dimension smaller than `min_diam`
        - remove blobs where more than `bcg_th` of the corresponding 
            bounding box pixels proportion are background.
    """
    clean_mask = np.zeros(mask.shape, dtype=bool)
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    for candidate_id, stat in enumerate(stats[1:], start=1):
        w, h = stat[2:4]
        if is_small_calcification(w, h, min_diam):
            continue
        # if is_blob_near_outline(image, stat, bcg_th):
        #     continue
        clean_mask = np.logical_or(clean_mask, labels==candidate_id)
    return clean_mask

def is_small_calcification(bb_width, bb_height, min_diam):
    """ remove blobs with greatest dimension smaller than `min_diam`
    """
    if max(bb_width, bb_height) < min_diam:
        return True
    return False

def is_blob_near_outline(image, bb, bcg_th=0.3):
    """ remove blobs where more than `bcg_th` of the corresponding 
    bounding box pixels proportion are background.
    TODO: remove blobs that has more than `th` pixels nearby the background
    """
    w, h = bb[2], bb[3]
    x1, x2, y1, y2 = bb[0], bb[0]+w, bb[1], bb[1]+h
    bb_area = w * h
    background_area = bb_area - np.count_nonzero(image[x1:x2, y1:y2])
    if background_area >= bb_area * bcg_th:
        return True
    return False

def get_centroids_mask(mask):
    centroids_mask = np.zeros(mask.shape)
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    for stat in stats[1:]:
        mean_y = round(stat[0] + stat[2]/2)
        mean_x = round(stat[1] + stat[3]/2)
        centroids_mask[mean_x, mean_y] = 1

    return centroids_mask

def preprocess_mamm(mamm, outline_erosion_diam=100, artifact_lower_t=0.8, artifact_left_w=0.1, artifact_min_area=2000):
    mamm = left_mamm(mamm)
    mamm = clean_mamm(mamm)
    act_w = get_act_width(mamm)  # identify column where background starts
    mamm = cut_mamm(mamm, act_w, first_n_col_to_drop=0)
    # erode to remove breast outline
    # mamm = remove_outline(mamm, erosion_diam=int(outline_erosion_diam))
    act_w = get_act_width(mamm)  # cut again
    mamm = cut_mamm(mamm, act_w, first_n_col_to_drop=0)

    mamm = remove_border_artifacts(mamm, artifact_lower_t, artifact_left_w, artifact_min_area)

    return mamm

def left_mamm(mamm):
    if mamm[:, :200, ...].sum() < mamm[:, -200:, ...].sum():
            mamm[:, :, ...] = mamm[:, ::-1, ...]

    return mamm

def get_act_width(mamm):
    w = mamm.shape[1] // 3

    while mamm[:, w:].max() > 0 and w < mamm.shape[1]-1:
        w += 1

    return w

def cut_mamm(mamm, act_w, first_n_col_to_drop=0):
    h = mamm.shape[0]
        # mamm[k] = v[:h - (h % 16), :act_w + (-act_w % 16)]
    mamm = mamm[:h, first_n_col_to_drop:act_w]

    # assert mamm['mamm'].shape[0] % 16 == mamm['mamm'].shape[1] % 16 == 0

    return mamm

def clean_outline(mamm, erosion_size=100):
    background_val = 0
    msk = (mamm > background_val).astype(np.uint8)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    # msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)))
    
    comps = cv2.connectedComponentsWithStats(msk)
    # Find largest area, discard others
    common_label = np.argmax(comps[2][1:, cv2.CC_STAT_AREA]) + 1

    msk = (comps[1] == common_label).astype(bool)

    mamm[:, :] = msk * mamm[:, :]

    # erosion_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(erosion_diam, erosion_diam))
    # eroded_img = cv2.erode(mamm, erosion_struct)
    # erosion_mask = (eroded_img > 0).astype(bool)
    # mamm[:, :] = erosion_mask * mamm[:, :]
    
    return mamm

def clean_mamm(mamm):
    background_val = 0
    # If three channels image, transform to grayscale (2d) buy only if the 3 channels
    # have the same intensity for each given pixel else turn pixel off
    if len(mamm.shape)==3:
        msk1 = (mamm[:, :, 0] == mamm[:, :, 1]) & (mamm[:, :, 1] == mamm[:, :, 2])
        mamm = mamm.mean(axis=2) * msk1
        
    msk = (mamm > background_val).astype(np.uint8)
    msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50)))

    comps = cv2.connectedComponentsWithStats(msk)
    # Find largest area, discard others
    common_label = np.argmax(comps[2][1:, cv2.CC_STAT_AREA]) + 1

    msk = (comps[1] == common_label).astype(bool)

    mamm[:, :] = msk * mamm[:, :]

    return mamm

def remove_outline(mamm, erosion_diam=100):
    """
    "We apply morphological erosion to a structure element radius of
    `erosion_diam`/2 --> 100 pixels to remove the pixels close to the breast outline."
    """
    erosion_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(erosion_diam, erosion_diam))
    eroded_img = cv2.erode(mamm, erosion_struct)
    erosion_mask = (eroded_img > 0).astype(bool)
    mamm[:, :] = erosion_mask * mamm[:, :]
    
    return mamm

def remove_border_artifacts(mamm, th=0.7, w=0.1, min_area=1000):
    """ Remove bright areas near the left border within `w` proportion of image width 
    t float lower bound for color intensity (fraction between 0 and 1 from maximum image intensity)
    w is the distance from left border within which to remove bright spots greater than 
    `min_area` pixels
    """
    max_color = np.max(mamm)
    th = int(max_color*th)
    w = round(mamm.shape[1] * w)

    artifact_mask = cv2.threshold(mamm[:, :w, ...], th, max_color, cv2.THRESH_BINARY)[1]
    artifact_mask = artifact_mask.astype(np.uint8)

    # first (quick??) check, total selected area is small
    if np.sum(artifact_mask) < min_area:
        return mamm

    comps = cv2.connectedComponentsWithStats(artifact_mask)
    mask = np.zeros(mamm[:, :w].shape).astype(bool)
    
    for label_idx, area in enumerate(comps[2][:, cv2.CC_STAT_AREA]):
        if area < min_area or area >= mamm.shape[0]*mamm.shape[1]:
            continue
        m = (comps[1] == label_idx).astype(bool)
        # each single area should be greater than `min_area`
        mi = mamm[:, :w]*m
        mis = mi>=th
        if np.count_nonzero(mis) < min_area:
            continue
        # Should be attached to the border -> 
        # 1/50th of all pixel should be in the first two columns
        if np.sum(m[:, :2]) < np.sum(m)/50:
            continue
        mask = np.logical_or(mask, m)#(mask + m).astype(bool)
    
    mask = mask.astype(np.uint8)

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(10, 10)))  # TODO
    # discard from image the identified portions 
    mamm[:, :w] = (1 - mask) * mamm[:, :w]

    return mamm


# def segment_collection(maps, overwrite=False):
#     """ turns mammografies into preprocessed pngs ready for segmentation
#         returns dataframe of mapping orig->preprocessed.png
#     """
#     rerun_preprocessing = False
#     outline_erosion_diam_mm = 10
#     calcification_min_diam_mm = 0.2
    

#     th = 0.5  # probability threshold
#     model = get_model(tag='calc_detect')

#     for input_fp, output_png in tqdm(maps.items()):
#         raw_pred_fp = output_png.replace('.png', '__mask_raw.png')
#         segmented_mask_fp = output_png.replace('.png', '__mask.png')
#         pixel_spacing, ps_y = pydicom.dcmread(input_fp, stop_before_pixels=True).ImagerPixelSpacing
#         calcification_min_diam = round(calcification_min_diam_mm/pixel_spacing)

#         mamm = preprocess_for_segmentation(input_fp, output_png, outline_erosion_diam_mm, rerun_preprocessing)

#         if not overwrite and os.path.exists(raw_pred_fp):
#             pred_prob = cv2.imread(raw_pred_fp, cv2.IMREAD_GRAYSCALE)/255
#         else:
#             pred_prob = predict(model, mamm)
#         pred_mask = np.uint8(pred_prob > th)

#         mask = discard_non_calcifications(mamm, pred_mask, calcification_min_diam, )

#         centroids_mask = get_centroids_mask(mask)



#         cv2.imwrite(output_png, mamm)
#         cv2.imwrite(raw_pred_fp, np.uint8(pred_prob*(2**8)))
#         # cv2.imwrite(segmented_mask_fp, np.uint8(mask*(2**8)))
#     return

# def parse_dicom_attributes(dcm: pydicom.dataset.FileDataset):
#     r = dict()
#     for attr in dir(dcm):
#         if not attr[0].isupper():
#             continue
#         if attr == 'PixelData':
#             continue
#         v = dcm.get(attr)
#         if isinstance(v, str):
#             try:
#                 v = int(v)
#             except ValueError as e1:
#                 try:
#                     v = float(v)
#                 except ValueError as e:
#                     pass
#         r[attr] = v
#     return r


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # preprocessed_reference_fp = os.path.join(project_dir, 'references', 'preprocessed_files_list.csv')

    main()
