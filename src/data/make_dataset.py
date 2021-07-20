# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
# if not dir1 in sys.path: sys.path.append(dir1)
from src.data.tcia_client import TCIAClient, printServerResponse
from src.data.utils import identify_bbox, get_png_filename, dicom_to_png
import json
import pandas as pd
import numpy as np
import urllib
import zipfile
from tqdm import tqdm
from glob import glob
import pydicom
# import cv2


# N_PATIENTS=00  # if zero download all patients
SEED=42  # to download the same N_patients
CLINICAL_DATA_URL='https://wiki.cancerimagingarchive.net/download/attachments/70230508/CMMD_clinicaldata_revision.xlsx?api=v2'
PIDS_TO_CHECK=['D1-0016', 'D1-0037', 'D1-0041', 'D1-0051', 'D1-0063', 'D1-0085', 'D1-0089', 'D1-0102', 'D1-0121', 'D1-0129', 'D1-0142', 'D1-0181', 'D1-0203', 'D1-0213', 'D1-0215', 'D1-0217', 'D1-0232', 'D2-0013', 'D2-0022', 'D2-0033', 'D2-0034', 'D2-0038', 'D2-0040', 'D2-0046', 'D2-0056', 'D2-0060', 'D2-0062', 'D2-0075', 'D2-0086']


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('N_PATIENTS', '-n', default=0, show_default=True, type=click.IntRange(min=0, max=1775, clamp=True))
def main(input_filepath, output_filepath, N_PATIENTS):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.setLevel('DEBUG')
    logger.info('making final data set from raw data')

    # CSV of dicom attributes extracted from collection
    dicom_attributes_fn = 'dicom_attributes.csv' if not N_PATIENTS else f'dicom_attributes-{N_PATIENTS}_seed-{SEED}.csv'
    dicom_attributes_fp = os.path.join(project_dir, 'data', 'processed', dicom_attributes_fn)

    logger.info(f'retrieving/reading series data from {series_fp}')
    series_df = get_series_df(series_fp)

    patientid_col = [c for c in series_df.columns if 'patientname' in c.lower() or 'patientid' in c.lower()]
    logger.debug(f'found {len(patientid_col)} `patient_name` column(s), using {patientid_col[0]}')
    patientid_col = patientid_col[0]

    if N_PATIENTS:
        np.random.seed(SEED)
        pids_to_download = np.random.choice(series_df[patientid_col].unique(), N_PATIENTS, replace=False)
        series_df = series_df.loc[series_df[patientid_col].isin(pids_to_download)]
    logger.info(f'ready to download {len(series_df)} series to {input_filepath}')
    download_collection(series_df, output_basedir=input_filepath, patient_col=patientid_col)

    # List all available dicoms
    dicom_fps_ = glob(os.path.join(input_filepath, '*/*.dcm'))
    # Filter only those downloaded in this run (folder name equals series instance id)
    dicom_fps = [fp for fp in dicom_fps_ if fp.split(os.path.sep)[-2] in series_df['SeriesInstanceUID'].values]
    logger.info(f'found {len(dicom_fps_)} dicoms, using {len(dicom_fps)}')

    labels_df = pd.read_excel(CLINICAL_DATA_URL, keep_default_na=False)
    logger.info(f'clinical data has {labels_df.shape} rows,columns with {labels_df.groupby(["ID1"]).size().shape[0]} patients')

    if not os.path.exists(dicom_attributes_fp):
        raw_attributes_fn = dicom_attributes_fn.replace('.csv', '_raw.csv')
        raw_attributes_fp = os.path.join(project_dir, 'data', 'interim', raw_attributes_fn)
        if not os.path.exists(raw_attributes_fp):
            logger.info('creating raw dicom attributes csv file')
            attrs = dict()
            for dicom_fp in tqdm(dicom_fps):
                attrs[dicom_fp] = get_dicom_attributes(dicom_fp)
            attrs = pd.DataFrame(attrs).T
            attrs.index.name = 'filepath'
            attrs.to_csv(raw_attributes_fp)
        df = pd.read_csv(raw_attributes_fp)
        logger.info('preprocessing raw dicom attributes')
        df, meta = preprocess_dicom_attributes_df(df)
        logger.debug(meta)
        df.to_csv(dicom_attributes_fp, index=False)
    dicom_attributes = pd.read_csv(dicom_attributes_fp)

    # Merge labels with dicom attrs
    df = pd.merge(labels_df, dicom_attributes, left_on=['ID1', 'LeftRight'], right_on=['PatientName', 'ImageLaterality'])
    logger.info(f'only {df.shape[0]} out of {dicom_attributes.shape[0]} images have a label assigned')
    df_fp = os.path.join(output_filepath, 'collection_details.csv')
    logger.info(f'writing merged labels+attributes to {df_fp}')
    df.to_csv(df_fp, index=False)

    # Convert DICOM to PNG
    dest_dir = os.path.join(project_dir, 'data', 'interim', 'collection')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    logger.info(f'converting {df.shape[0]} dicom images to png in folder {dest_dir}')
    for idx, row in tqdm(df.iterrows()):
        fp = row['filepath']
        png_fn = get_png_filename(row)
        png_fp = os.path.join(dest_dir, png_fn)

        dicom_to_png(fp, png_fp, overwrite=False)

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

def download_collection(df, output_basedir, patient_col=None, remove_zip=True):
    patient_col = patient_col or 'PatientID'
    for _, row in tqdm(df.iterrows()):
        seriesuid = row['SeriesInstanceUID']
        get_image(seriesuid, dest_basedir=output_basedir, remove_zip=remove_zip)
    return

def get_series_df(series_fp):
    """ Obtain series details from TCIA and write them to file  
    return pd.DataFrame
    """
    # seriesUID attribute is needed to download basically everything else
    if not os.path.exists(series_fp):
        try:
            response = tcia_client.get_series(collection = 'CMMD', outputFormat='json')
            response = json.loads(response.read().decode(response.info().get_content_charset('utf8')))
        except urllib.error.HTTPError as err:
            print ("Error executing " + tcia_client.GET_SERIES + ":\nError Code: ", str(err.code) , "\nMessage: " , err.read())
        # write to file
        pd.DataFrame.from_dict(response).to_csv(series_fp, index=False)
    df = pd.read_csv(series_fp)
    return df

def get_image(seriesuid, dest_basedir=None, remove_zip=True):
    dest_basedir = dest_basedir or os.path.join(project_dir, 'data', 'raw')
    dest_dir = os.path.join(dest_basedir, seriesuid)
    zip_fn = f'{seriesuid}.zip'
    zip_fp = os.path.join(dest_basedir, zip_fn)

    # check if folder exists and is not empty
    # TODO: check for dicoms (at least a couple file or multiple of 2, always in CMMD coll.)
    if os.path.exists(dest_dir) and os.path.isdir(dest_dir) and os.listdir(dest_dir):
        return dest_dir

    if not os.path.exists(zip_fp):
        tcia_client.get_image(seriesuid, dest_basedir, zip_fn)
    else:
        try:
            zipfile.ZipFile(zip_fp)
        except (IOError, zipfile.BadZipfile) as e:
            print('Bad zip file given as input.  (%s) %s' % (zip_fp, e))
            os.remove(zip_fp)
            tcia_client.get_image(seriesuid, dest_basedir, zip_fn)

    with zipfile.ZipFile(zip_fp, 'r') as zp:
        zp.extractall(dest_dir)

    if remove_zip:
        os.remove(zip_fp)
    return dest_dir

def get_dicom_attributes(dcm_fp):
    r = dict()
    dcm = pydicom.dcmread(dcm_fp, stop_before_pixels=True)
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

def preprocess_dicom_attributes_df(df):
    df['AcquisitionDate'] = pd.to_datetime(df['AcquisitionDate'], format='%Y%m%d')
    df['ContentDate'] = pd.to_datetime(df['ContentDate'], format='%Y%m%d')
    df['SeriesDate'] = pd.to_datetime(df['SeriesDate'], format='%Y%m%d')
    df['StudyDate'] = pd.to_datetime(df['StudyDate'], format='%Y%m%d')
    df['PatientAge'] = df['PatientAge'].str.strip('Y').astype(int)
    
    meta = list()  # dict()  # metadata
    for c in df.columns:
        uq = df[c].unique()
        if len(uq)==1:
            meta_ = f'Dropped column "{c}" because only a single value="{uq[0]}" is present'
            # meta[c] = uq[0]
            meta.append(meta_)
            df.drop(c, inplace=True, axis=1)
    return df, meta

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # secret_env = dotenv_values('.env')

    tcia_key = os.environ.get('TCIA_API_KEY', 'tcia key not found!')
    baseurl = os.environ.get('BASEURL', 'https://services.cancerimagingarchive.net/services/v4/TCIA/query')
    tcia_client = TCIAClient(credentials = tcia_key, baseUrl = baseurl)

    # print("There are {} CPUs on this machine ".format(cpu_count()))
    # pool = Pool(cpu_count()-1)
    # download_func = partial(download_zip, filePath = filePath)
    # results = pool.map(download_func, urls)
    # pool.close()
    # pool.join()

    # series path used to retrieve DICOM images using SeriesInstanceUID attribute
    series_fp = os.path.join(project_dir, 'data', 'external', 'series.csv')

    bboxes_dir = os.path.join(project_dir, 'data', 'processed', 'bboxes')
    masks_dir  = bboxes_dir.replace('bboxes', 'masks')
    if not os.path.exists(bboxes_dir) or not os.path.exists(masks_dir):
        os.makedirs(bboxes_dir)
        os.makedirs(masks_dir)

    main()
