# -*- coding: utf-8 -*-
import collections
from genericpath import exists
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from src.data.tcia import TCIAClient
from src.data.utils import identify_bbox, get_png_filename, dicom_to_png
import json
import pandas as pd
import numpy as np
import cv2
import pydicom
import urllib
import zipfile
from tqdm import tqdm
from glob import glob
from collections import defaultdict


# N_PATIENTS=00  # if zero download all patients
SEED=42  # to download the same N_patients
CLINICAL_DATA_URL='https://wiki.cancerimagingarchive.net/download/attachments/70230508/CMMD_clinicaldata_revision.xlsx?api=v2'

@click.command()
@click.argument('collection_reference_fp', )  #type=click.Path(exists=False))
@click.argument('output_dir', type=click.Path(exists=False))
@click.option('N_PATIENTS', '-n', default=0, show_default=True, type=click.IntRange(min=0, max=1775, clamp=True))
def main(collection_reference_fp, output_dir, N_PATIENTS):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    # collection_reference_fp is located in the input_data_dir
    # this info is needed to be agnostic to the folder structure
    input_dir = os.path.sep.join(collection_reference_fp.split(os.path.sep)[:-1])
    preprocessed_reference_fp = os.path.join(output_dir, collection_reference_fp.split(os.path.sep)[-1])

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

    # Preprocess images
    logger.info(f'preparing folders for prepocessing')
    preprocessed_ref_df = map_preprocessing(collection_ref_df['filepath'], input_dir, output_dir)
    logger.info(f'preprocessing images, saving to {output_dir}')
    logger.info(f'preserving original input folder structure as in {input_dir}')
    preprocess_collection(preprocessed_ref_df, overwrite=False)
    logger.info(f'saving preprocessed filepath references to {preprocessed_reference_fp}')
    preprocessed_ref_df.to_csv(preprocessed_reference_fp)

    # labels_df = pd.read_excel(CLINICAL_DATA_URL, keep_default_na=False)
    # logger.info(f'clinical data has {labels_df.shape} rows,columns with {labels_df.groupby(["ID1"]).size().shape[0]} patients')

    # # CSV of dicom attributes extracted from collection
    dicom_attributes_fn = 'dicom_attributes.csv' if not N_PATIENTS else f'dicom_attributes-{N_PATIENTS}_seed-{SEED}.csv'
    dicom_attributes_fp = os.path.join(project_dir, 'data', 'raw', dicom_attributes_fn)

    # labels_df = pd.read_excel(CLINICAL_DATA_URL, keep_default_na=False)
    # logger.info(f'clinical data has {labels_df.shape} rows,columns with {labels_df.groupby(["ID1"]).size().shape[0]} patients')

    # # List all available dicoms
    # all_dicom_fps = glob(os.path.join(collection_dir, '*/*.dcm'))
    # # Filter by patient id (folder name equals series instance id)
    # dicom_fps = [fp for fp in all_dicom_fps if fp.split(os.path.sep)[-2] in series_df['SeriesInstanceUID'].values]
    # logger.info(f'found {len(all_dicom_fps)} dicoms, using {len(dicom_fps)}')

    # if not os.path.exists(dicom_attributes_fp):
    #     raw_attributes_fn = dicom_attributes_fn.replace('.csv', '_raw.csv')
    #     raw_attributes_fp = os.path.join(project_dir, 'data', 'interim', raw_attributes_fn)
    #     logger.info('extracting and saving to csv dicom attributes')
    #     # raw_dicom_df = get_dicom_attributes(dicom_fps)
    #     df = dicoms_to_df(dicom_fps)
    #     if clean_dicom_metadata:
    #         logger.info('extracting dicom attributes and saving to df\tcleaning dates and removing zero variance cols :)')
    #         df, meta = preprocess_dicom_attributes_df(df)
    #         logger.debug(meta)
    #     df.to_csv(dicom_attributes_fp, index=False)
    # dicom_attributes = pd.read_csv(dicom_attributes_fp, keep_default_na='')

    # # Merge labels with dicom attrs
    # df = pd.merge(labels_df, dicom_attributes, left_on=['ID1', 'LeftRight'], right_on=['PatientName', 'ImageLaterality'])
    # logger.info(f'only {df.shape[0]} out of {dicom_attributes.shape[0]} images have a label assigned')
    # df_fp = os.path.join(output_dir, 'collection_details.csv')
    # logger.info(f'writing merged labels+attributes to {df_fp}')
    # df.to_csv(df_fp, index=False)

    # # Convert DICOM to PNG
    # dest_dir = os.path.join(project_dir, 'data', 'interim', 'collection')
    # if not os.path.exists(dest_dir):
    #     os.makedirs(dest_dir)
    # logger.info(f'converting {df.shape[0]} dicom images to png in folder {dest_dir}')
    # for idx, row in tqdm(df.iterrows()):
    #     fp = row['filepath']
    #     png_fn = get_png_filename(row)
    #     png_fp = os.path.join(dest_dir, png_fn)

    #     dicom_to_png(fp, png_fp, overwrite=False)

    return

    # logger.info(f'identifying bounding boxes with erosion + tophat + otsu + dilatation + max. area')
    # for _, row in tqdm(df.iterrows()):
    #     dcm_fp = row['filepath']
    #     fname = get_png_filename(row)
    #     bbox_fp = os.path.join(bboxes_dir, fname)
    #     mask_fp = os.path.join(masks_dir, fname)
    #     mask, bb_img = get_boundingbox(dcm_fp)
    #     if bb_img is None:
    #         logger.warning(f'failed to compute bounding box for ', dcm_fp)
    #         continue
    #     # Write to file
    #     cv2.imwrite(mask_fp, mask * 255)
    #     cv2.imwrite(bbox_fp, bb_img)

def map_preprocessing(filepaths, input_dir, output_root_dir):
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

def parse_dicom_attributes(dcm: pydicom.dataset.FileDataset):
    r = dict()
    for attr in dir(dcm):
        if not attr[0].isupper():
            continue
        if attr == 'PixelData':
            continue
        v = dcm.get(attr)
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError as e1:
                try:
                    v = float(v)
                except ValueError as e:
                    pass
        r[attr] = v
    return r

def preprocess_collection(maps, overwrite=False):
    """ turns mammografies into preprocessed pngs ready for segmentation
        returns dataframe of mapping orig->preprocessed.png
    """
    for input_fp, output_png in tqdm(maps.items()):
        if not overwrite and os.path.exists(output_png):
            continue
        if input_fp.endswith('dcm'):
            dicom = pydicom.dcmread(input_fp)
            mamm = dicom.pixel_array
        else:
            mamm = cv2.imread(input_fp, cv2.IMREAD_UNCHANGED)
        mamm = preprocess_mamm(mamm, outline_erosion_diam=150)
        cv2.imwrite(output_png, mamm)
        # images_metadata[input_fp] = metadata_
    return
    
def preprocess_mamm(mamm, outline_erosion_diam=200, artifact_lower_t=0.8, artifact_left_w=0.1, artifact_min_area=2000):
    # put it on the left
    mamm = left_mamm(mamm)
    mamm = clean_mamm(mamm)
    # identify column where background starts
    act_w = get_act_width(mamm)
    mamm = cut_mamm(mamm, act_w, first_n_col_to_drop=0)
    # erode to remove breast outline
    mamm = remove_outline(mamm, erosion_diam=outline_erosion_diam)
    # cut again
    act_w = get_act_width(mamm)
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

def clean_mamm(mamm):
    background_val = 0
    # mamm[:10, :, ...] = 0
    # mamm[-10:, :, ...] = 0
    # mamm[:, -10:, ...] = 0
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

def remove_outline(mamm, erosion_diam=200):
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
    # print(artifact_mask.shape, artifact_mask.dtype)

    # mask = np.zeros(mamm.shape).astype(bool)
    # mask[:, :w] = artifact_mask
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




def get_boundingbox(dcm_fp, erosion_size=None, tophat_size=None, dilatation_size=None):
    erosion_size = erosion_size or (200,200)
    tophat_size = tophat_size or (200,16)
    dilatation_size = dilatation_size or (120,120)

    image = pydicom.dcmread(dcm_fp).pixel_array.astype(np.uint8)    
    mask = np.zeros(image.shape, dtype=bool)
    bb = None

    roi = identify_bbox(image, erosion_size, tophat_size, dilatation_size)
    if roi is not None:
        mask[roi[2]:roi[3], roi[0]:roi[1]] = 1
        bb = image[roi[2]:roi[3], roi[0]:roi[1]]
        # cv2.imwrite(mask_fp, mask * 255)
        # cv2.imwrite(bbox_fp, bb)
    return mask, bb




def get_dicom_attributes_df(fps):
    attrs = dict()
    for dicom_fp in tqdm(fps):
        attrs[dicom_fp] = get_dicom_attributes(dicom_fp)
    df = pd.DataFrame(attrs).T
    df.index.name = 'filepath'
    return df
def get_dicom_attributes(fp):
    r = dict()
    dcm = pydicom.dcmread(fp, stop_before_pixels=True)
    for attr in dir(dcm):
        if not attr[0].isupper():
            continue
        v = dcm.get(attr)
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError as e1:
                try:
                    v = float(v)
                except ValueError as e:
                    pass
        r[attr] = v
    return r

def preprocess_raw_dicom_attributes(df):
    df['AcquisitionDate'] = pd.to_datetime(df['AcquisitionDate'], format='%Y%m%d')
    df['ContentDate'] = pd.to_datetime(df['ContentDate'], format='%Y%m%d')
    df['SeriesDate'] = pd.to_datetime(df['SeriesDate'], format='%Y%m%d')
    df['StudyDate'] = pd.to_datetime(df['StudyDate'], format='%Y%m%d')
    df['PatientAge'] = df['PatientAge'].str.strip('Y').astype(int)
    return df

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # collection_reference_fp = os.path.join(project_dir, 'references', 'collection_files_list.csv')
    # preprocessed_reference_fp = os.path.join(project_dir, 'references', 'preprocessed_files_list.csv')

    main()
