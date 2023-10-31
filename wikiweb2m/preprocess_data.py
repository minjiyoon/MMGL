import json
import time
import glob
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
from collections import defaultdict

import requests


def sparse_tf_to_numpy(tf_sparse_tensor):
    """
    Converts a tf.SparseTensor to a list of numpy object.
    Args: 
        tf_sparse_tensor: A tf.SparseTensor object.
    Returns:
        A list of numpy object.
    """
    array = tf.sparse.to_dense(tf_sparse_tensor).numpy()
    if len(array.shape) == 2:
        array = array.reshape(-1)
    return array.tolist()

def convert_to_numpy(page_id, d):
    """
    Converts a tf.Tensor to a list of numpy object.
    Args:
        page_id: A page id.
        d: A tf.Tensor object.
    Returns:
        A list of numpy object.        
    """
    page_url = d[0]['page_url'].numpy()
    page_title = d[0]['page_title'].numpy()
    page_description = d[0]['clean_page_description'].numpy()
    section_title = sparse_tf_to_numpy(d[1]['section_title'])
    section_depth = sparse_tf_to_numpy(d[1]['section_depth'])
    section_heading = sparse_tf_to_numpy(d[1]['section_heading_level'])
    section_parent_index = sparse_tf_to_numpy(d[1]['section_parent_index'])
    section_summary = sparse_tf_to_numpy(d[1]['section_clean_1st_sentence'])
    section_rest_sentence = sparse_tf_to_numpy(d[1]['section_rest_sentence'])
    image_url =  sparse_tf_to_numpy(d[1]['section_image_url'])
    image_caption = sparse_tf_to_numpy(d[1]['section_image_captions'])

    return [page_id, page_url, page_title, page_description, section_title, section_depth, section_heading, \
                section_parent_index, section_summary, section_rest_sentence, image_url, image_caption]

class DataParser():
    """
    Parses the tfrecord files and saves the data as parquet files.
    Follow the WikiWeb2M dataset format (https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md).
    """
    def __init__(self):
        self.path = './wikiweb2m/raw/'
        self.filepath = 'wikiweb2m-*'
        self.suffix = '.tfrecord*'
        self.parse_data()

    def parse_data(self):
        context_feature_description = {
            'split': tf.io.FixedLenFeature([], dtype=tf.string),
            'page_title': tf.io.FixedLenFeature([], dtype=tf.string),
            'page_url': tf.io.FixedLenFeature([], dtype=tf.string),
            'clean_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
            'raw_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
            'is_page_description_sample': tf.io.FixedLenFeature([], dtype=tf.int64),
            'page_contains_images': tf.io.FixedLenFeature([], dtype=tf.int64),
            'page_content_sections_without_table_list': tf.io.FixedLenFeature([] , dtype=tf.int64)
        }

        sequence_feature_description = {
            'is_section_summarization_sample': tf.io.VarLenFeature(dtype=tf.int64),
            'section_title': tf.io.VarLenFeature(dtype=tf.string),
            'section_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_depth': tf.io.VarLenFeature(dtype=tf.int64),
            'section_heading_level': tf.io.VarLenFeature(dtype=tf.int64),
            'section_subsection_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_parent_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_text': tf.io.VarLenFeature(dtype=tf.string),
            'section_clean_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'section_raw_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'section_rest_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'is_image_caption_sample': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_url': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_mime_type': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_width': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_height': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_in_wit': tf.io.VarLenFeature(dtype=tf.int64),
            'section_contains_table_or_list': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_captions': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_alt_text': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_raw_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_clean_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_raw_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_clean_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_contains_images': tf.io.VarLenFeature(dtype=tf.int64)
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_sequence_example(example_proto,
                                                        context_feature_description,
                                                        sequence_feature_description)

        data_path = glob.glob(self.path + self.filepath + self.suffix)
        raw_dataset = tf.data.TFRecordDataset(data_path, compression_type='GZIP')
        self.dataset = raw_dataset.map(_parse_function)

    def save_df_torch(self):
        # save as parquet files

        # columns describing each section
        columns = ['page_id', 'page_url', 'page_title', 'page_description', 'section_title', 'section_depth', 'section_heading', \
                'section_parent_index', 'section_summary', 'section_rest_sentence', 'image_url', 'image_caption']

        train_df = pd.DataFrame(columns=columns)
        val_df = pd.DataFrame(columns=columns)
        test_df = pd.DataFrame(columns=columns)

        for page_id, d in enumerate(self.dataset):
            if page_id % 100000 == 0:
                print(page_id, 'have processed...')
            # we sample first 600k pages
            if page_id == 600000:
                break
            split = d[0]['split'].numpy().decode()
            # we sample first 400k pages for training, next 100k for validation, and next 100k for testing
            if page_id < 400000:
                train_df.loc[len(train_df)] = convert_to_numpy(page_id, d)
            elif page_id < 500000:
                val_df.loc[len(val_df)] = convert_to_numpy(page_id, d)
            else:
                test_df.loc[len(test_df)] = convert_to_numpy(page_id, d)

        print(f'train_num: ', len(train_df), ', val_num: ', len(val_df), ', test_num: ', len(test_df))
        train_df.to_parquet(f'{self.path}/wikiweb2m_train_large.parquet')
        val_df.to_parquet(f'{self.path}/wikiweb2m_val_large.parquet')
        test_df.to_parquet(f'{self.path}/wikiweb2m_test_large.parquet')

    def split_ids(self, task):
        # split page ids into training/validation/test sets and save as pickle files
        id_list = defaultdict(list)
        for page_id, d in enumerate(self.dataset):
            if page_id % 100000 == 0:
                print(page_id, 'have processed...')
            # we sample first 600k pages
            if page_id == 600000:
                break
            # we sample first 400k pages for training, next 100k for validation, and next 100k for testing
            if page_id < 400000:
                split = "train"
            elif page_id < 500000:
                split = "val"
            else:
                split = "test"
            
            # when task is page summarization
            if task == 'page':
                is_sample = d[0]['is_page_description_sample'].numpy()
                if is_sample == 0:
                    continue
                id_list[split].append(page_id)
            # when task is section summarization
            elif task == 'section':
                are_samples = d[1]['is_section_summarization_sample'].values.numpy()
                for section_id in range(are_samples.shape[0]):
                    is_sample = are_samples[section_id]
                    if is_sample == 0:
                        continue
                    id_list[split].append((page_id, section_id))

        print(f'task: {task}, train_num: ', len(id_list['train']), ', val_num: ', len(id_list['val']), ', test_num: ', len(id_list['test']))
        with open(f'{self.path}/{task}_id_split_large.pkl', 'wb') as file:
            pickle.dump(id_list, file)

    def download_images(self):
        # download images from image urls

        headers = {"User-Agent": "research (https://www.cs.cmu.edu/; minjiy@cs.cmu.edu)"}

        for page_id, d in enumerate(self.dataset):
            # we sample first 600k pages
            if page_id < 600000:
                continue
            if page_id % 1000 == 0:
                print(page_id, 'have processed...')
            image_urls = tf.sparse.to_dense(d[1]['section_image_url']).numpy()
            for section_id in range(image_urls.shape[0]):
                for image_id in range(image_urls[section_id].shape[0]):
                    image_url = image_urls[section_id][image_id]
                    if image_url == b'':
                        continue
                    image_url = image_url.decode()
                    file_format = os.path.splitext(image_url)[1][1:]
                    file_name = f'{self.path}/images/{page_id}_{section_id}_{image_id}.{file_format}'
                    if os.path.exists(file_name):
                        break

                    another_image = False
                    try:
                        response = requests.get(image_url, headers=headers)
                        response.raise_for_status()
                    except requests.exceptions.HTTPError as e:
                        if "404 Client Error: Not Found for url" in str(e):
                            # corresponding image does not exist
                            another_image = True
                            continue
                        else:
                            # Wikimedia server is busy; try again after 1 second
                            time.sleep(1)
                            response = requests.get(image_url)

                    with open(file_name, 'wb') as file:
                        for chunk in response.iter_content(8192):
                            file.write(chunk)
                    # check if the downloaded file is a right format
                    try:
                        img = Image.open(file_name)
                    except:
                        if os.path.exists(file_name):
                            os.remove(file_name)
                        another_image = True
                        continue
                    # if another_image == True, we try to download another image in the same section
                    if another_image == False:
                        break


if __name__ == "__main__":
    parser = DataParser()
    # split (page ids, section_ids) into training/validation/test sets and save as pickle files
    parser.split_ids('section')
    # save WikiWeb2M data as parquet files
    parser.save_df_torch()
    # download images from image urls
    parser.download_images()
