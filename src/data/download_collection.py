# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
from dotenv import find_dotenv, load_dotenv
import json
import pandas as pd
import numpy as np
import urllib
import zipfile
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from src.data.tcia import TCIAClient


# N_PATIENTS=00  # if zero download all patients
SEED=42  # to download the same N_patients
CLINICAL_DATA_URL='https://wiki.cancerimagingarchive.net/download/attachments/70230508/CMMD_clinicaldata_revision.xlsx?api=v2'


@click.command()
@click.argument('collection_dir', type=click.Path())
@click.argument('collection_reference_filename', type=click.STRING, default='collection_files_list.csv')
@click.option('N_PATIENTS', '-n', default=0, show_default=True, type=click.IntRange(min=0, max=1775, clamp=True))
def main(collection_dir, collection_reference_filename, N_PATIENTS):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    collection_reference_fp = os.path.join(collection_dir, collection_reference_filename)
    if os.path.exists(collection_reference_fp):
        logger.info(f'output file {collection_reference_fp} already exists, skipping')
        return pd.read_csv(collection_reference_fp)

    if os.path.exists(series_fp):
        logger.info(f'getting collection series uids from {series_fp}')
        series_df = pd.read_csv(series_fp)
    else:
        logger.info(f'downloading collection series uids data from TCIA')
        series_df = get_series_uids(series_fp)
    logger.debug(f'showing 3 samples out of {len(series_df)} from {series_fp}...')
    logger.debug(series_df.sample(3, random_state=SEED, replace=False))

    # Filter by patient id
    # patient information is needed mainly for testing on a small number of samples
    if N_PATIENTS:
        np.random.seed(SEED)
        patientid_cols = [c for c in series_df.columns if 'patientname' in c.lower() or 'patientid' in c.lower()]
        patientid_col = patientid_cols[0]
        logger.debug(f'found {len(patientid_cols)} `patient_name` column(s), using {patientid_col}')
        selected_pids = np.random.choice(series_df[patientid_col].unique(), N_PATIENTS, replace=False)
        series_df = series_df.loc[series_df[patientid_col].isin(selected_pids)]
    logger.debug(f'number of filtered series uids: {len(series_df)}')

    logger.info(f'downloading collection images to {collection_dir}')
    if not os.path.exists(collection_dir):
        logger.info(f'creating directory {collection_dir} as it doesn\'t exist')
        os.makedirs(collection_dir, exist_ok=True)
    collection_ref_df = download_collection(series_df['SeriesInstanceUID'], collection_dir, remove_zip=True)
    collection_ref_df.to_csv(collection_reference_fp)
    logger.info(f'{len(collection_ref_df)} collection references saved to {collection_reference_fp}')
    logger.debug(f'3 samples:\n{collection_ref_df.sample(3, random_state=SEED, replace=False)}')

    return collection_ref_df

def get_series_uids(series_fp, collection = 'CMMD'):
    """ Obtain series details from TCIA and save to csv.  
        SeriesInstanceUID attribute is needed to download collection images
        
        return pd.DataFrame about series details
    """
    df = fetch_series_instance(collection)
    df.to_csv(series_fp, index=False)
    return pd.read_csv(series_fp)

def fetch_series_instance(collection):
    """Query TCIA for series data
    """
    try:
        response = tcia_client.get_series(collection = collection, outputFormat='json')
        response = json.loads(response.read().decode(response.info().get_content_charset('utf8')))
    except urllib.error.HTTPError as err:
        print ("Error executing " + tcia_client.GET_SERIES + ":\nError Code: ", str(err.code) , "\nMessage: " , err.read())
    return pd.DataFrame.from_dict(response)

def download_collection(series_uids, output_basedir, remove_zip=True):
    """ download (if necessary) series images from TCIA
    return dataframe of collected files for each series id
         dictionary {series_uid: [downloaded_files]} 
        mapping series_uid to downloaded images filepaths {'uid': ['1.dcm', '2.dcm', 'unexpected.foo']}
    """
    series_to_fps = defaultdict(list)
    for series_uid in tqdm(series_uids):
        series_dir = get_series(series_uid, dest_basedir=output_basedir, remove_zip=remove_zip)
        series_to_fps[series_uid] = glob(os.path.join(series_dir, '*'))  # list downloaded `everything` (ie .dcm)
    
    collection_ref_df = pd.DataFrame.from_dict(series_to_fps, orient='index')
    collection_ref_df.index.name = series_uids.name
    collection_ref_df = pd.DataFrame(collection_ref_df.unstack().dropna().sort_values(), columns = ['filepath'])
    collection_ref_df.index = collection_ref_df.index.droplevel(0)  # files order is not relevant

    return collection_ref_df

def get_series(seriesuid, dest_basedir, remove_zip=True):
    # arbitrary choice to put downloaded series images inside 
    # a folder named after sereis uid
    # if know a-priori dicom filenames are all unique in the collection
    # adding this subdirectory could be avoided  
    series_destdir = os.path.join(dest_basedir, seriesuid)
    # check if folder exists and is not empty, assumes unzipping won't fail!
    if os.path.exists(series_destdir) and os.path.isdir(series_destdir) and os.listdir(series_destdir):
        return series_destdir

    zip_fn = f'{seriesuid}.zip'
    zip_fp = os.path.join(dest_basedir, zip_fn)

    # if an archive is already there, use it
    if not os.path.exists(zip_fp):
        # This check assumes archive file names are unique (i.e. different) in the collection 
        # otherwise will always extract same archive
        tcia_client.get_image(seriesuid, dest_basedir, zip_fn)
    else: 
        # if something breaks during download archive will be corrupted
        # so that next time the program is run it will break when resuming from last patient 
        try:
            zipfile.ZipFile(zip_fp)
        except (IOError, zipfile.BadZipfile) as e:
            # remove archive and retry download
            print('Bad zip file given as input.  (%s) %s' % (zip_fp, e))
            # TODO: check if get_image endpoint can resume download from partial archives
            os.remove(zip_fp)
            tcia_client.get_image(seriesuid, dest_basedir, zip_fn)

    with zipfile.ZipFile(zip_fp, 'r') as zp:
        zp.extractall(series_destdir)
    if remove_zip:
        os.remove(zip_fp)

    return series_destdir


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # secret_env = dotenv_values('.env')

    tcia_key = os.environ.get('TCIA_API_KEY', 'tcia key not found!')
    baseurl = os.environ.get('BASEURL', 'https://services.cancerimagingarchive.net/services/v4/TCIA/query')
    tcia_client = TCIAClient(credentials = tcia_key, baseUrl = baseurl)

    # series path used to retrieve DICOM images using SeriesInstanceUID attribute
    series_fp = os.path.join(project_dir, 'data', 'external', 'series.csv')
    # clean_dicom_metadata = True  # whether to clean raw dicom attributes

    # collection_reference_fp = os.path.join(project_dir, 'references', 'collection_files_list.csv')
    # preprocessed_reference_fp = os.path.join(project_dir, 'references', 'preprocessed_files_list.csv')

    main()
