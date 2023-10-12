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

def convert_tf_to_scipy(tf_sparse_tensor):
    array = tf.sparse.to_dense(tf_sparse_tensor).numpy()
    if len(array.shape) == 2:
        array = array.reshape(-1)
    return array.tolist()

def convert_to_scipy(page_id, d):
    page_url = d[0]['page_url'].numpy()
    page_title = d[0]['page_title'].numpy()
    page_description = d[0]['clean_page_description'].numpy()
    section_title = convert_tf_to_scipy(d[1]['section_title'])
    section_depth = convert_tf_to_scipy(d[1]['section_depth'])
    section_heading = convert_tf_to_scipy(d[1]['section_heading_level'])
    section_parent_index = convert_tf_to_scipy(d[1]['section_parent_index'])
    section_summary = convert_tf_to_scipy(d[1]['section_clean_1st_sentence'])
    section_rest_sentence = convert_tf_to_scipy(d[1]['section_rest_sentence'])
    image_url =  convert_tf_to_scipy(d[1]['section_image_url'])
    image_caption = convert_tf_to_scipy(d[1]['section_image_captions'])

    return [page_id, page_url, page_title, page_description, section_title, section_depth, section_heading, \
                section_parent_index, section_summary, section_rest_sentence, image_url, image_caption]

class DataParser():
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
        columns = ['page_id', 'page_url', 'page_title', 'page_description', 'section_title', 'section_depth', 'section_heading', \
                'section_parent_index', 'section_summary', 'section_rest_sentence', 'image_url', 'image_caption']

        train_df = pd.DataFrame(columns=columns)
        val_df = pd.DataFrame(columns=columns)
        test_df = pd.DataFrame(columns=columns)

        for page_id, d in enumerate(self.dataset):
            if page_id % 100000 == 0:
                print(page_id, 'have processed...')
            if page_id == 600000:
                break
            split = d[0]['split'].numpy().decode()
            #if split == "train":
            if page_id < 400000:
                train_df.loc[len(train_df)] = convert_to_scipy(page_id, d)
            #elif split == "val":
            elif page_id < 500000:
                val_df.loc[len(val_df)] = convert_to_scipy(page_id, d)
            else:
                test_df.loc[len(test_df)] = convert_to_scipy(page_id, d)

        print(f'train_num: ', len(train_df), ', val_num: ', len(val_df), ', test_num: ', len(test_df))
        train_df.to_parquet(f'{self.path}/wikiweb2m_train_large.parquet')
        val_df.to_parquet(f'{self.path}/wikiweb2m_val_large.parquet')
        test_df.to_parquet(f'{self.path}/wikiweb2m_test_large.parquet')

    def save_list(self):
        #page_list = defaultdict(list)
        section_list = defaultdict(list)
        #image_list = defaultdict(list)
        for page_id, d in enumerate(self.dataset):
            if page_id % 100000 == 0:
                print(page_id, 'have processed...')
            split = d[0]['split'].numpy().decode()
            # page description task
            #is_sample = d[0]['is_page_description_sample'].numpy()
            #if is_sample == 1:
            #    page_list[split].append(d)
            # section summarization task
            are_samples = tf.sparse.to_dense(d[1]['is_section_summarization_sample']).numpy()
            for section_id in range(are_samples.shape[0]):
                is_sample = are_samples[section_id][0]
                if is_sample == 1:
                    section_list[split].append((section_id, d))
            # image summarization task
            #are_samples = tf.sparse.to_dense(d[1]['is_image_caption_sample']).numpy()
            #for section_id in range(are_samples.shape[0]):
            #    for image_id in range(are_samples[section_id].shape[0]):
            #        is_sample = are_samples[section_id][image_id]
            #        if is_sample == 1:
            #            image_list[split].append((section_id, image_id, d))

        #print(f'task: page, train_num: ', len(page_list['train']), ', val_num: ', len(page_list['val']), ', test_num: ', len(page_list['test']))
        print(f'task: section, train_num: ', len(section_list['train']), ', val_num: ', len(section_list['val']), ', test_num: ', len(section_list['test']))
        #print(f'task: image, train_num: ', len(image_list['train']), ', val_num: ', len(image_list['val']), ', test_num: ', len(image_list['test']))

        for split in ('train', 'val', 'test'):
            #with open(f'{self.path}/wikiweb2m_page_{split}.pkl', 'wb') as file:
            #    pickle.dump(page_list[split], file)
            with open(f'{self.path}/wikiweb2m_section_{split}_small.pkl', 'wb') as file:
                pickle.dump(section_list[split][:10000], file)
            #with open(f'{self.path}/wikiweb2m_section_{split}_medium.pkl', 'wb') as file:
            #    pickle.dump(section_list[split][:100000], file)
            #with open(f'{self.path}/wikiweb2m_section_{split}_large.pkl', 'wb') as file:
            #    pickle.dump(section_list[split][:1000000], file)
            #with open(f'{self.path}/wikiweb2m_image_{split}.pkl', 'wb') as file:
            #    pickle.dump(image_list[split], file)


    def split_preprocess(self):
        page_list = defaultdict(list)
        section_list = defaultdict(list)
        image_list = defaultdict(list)
        for page_id, d in enumerate(self.dataset):
            if page_id % 100000 == 0:
                print(page_id, 'have processed...')
            if page_id == 100000:
                break

            # page information
            data = {}
            data['page_url'] = d[0]['page_url'].numpy()
            data['page_title'] = d[0]['page_title'].numpy()
            data['clean_page_description'] = d[0]['clean_page_description'].numpy()
            # section information
            section_titles = tf.sparse.to_dense(d[1]['section_title'])
            for section_id in range(section_titles.shape[0]):
                data[f'section_title_{section_id}'] = section_titles[section_id][0].numpy()
                data[f'section_depth_{section_id}'] = tf.sparse.to_dense(d[1]['section_depth'])[section_id][0].numpy()
                data[f'section_heading_level_{section_id}'] = tf.sparse.to_dense(d[1]['section_heading_level'])[section_id][0].numpy()
                data[f'section_parent_index_{section_id}'] = tf.sparse.to_dense(d[1]['section_parent_index'])[section_id][0].numpy()
                data[f'section_clean_1st_sentence_{section_id}'] = tf.sparse.to_dense(d[1]['section_clean_1st_sentence'])[section_id][0].numpy()
                data[f'section_rest_sentence_{section_id}'] = tf.sparse.to_dense(d[1]['section_rest_sentence'])[section_id][0].numpy()
            # image information
            image_urls =  tf.sparse.to_dense(d[1]['section_image_url'])
            for section_id in range(image_urls.shape[0]):
                for image_id in range(image_urls[section_id].shape[0]):
                    if image_urls[section_id][image_id].numpy() == b'':
                        continue
                    data[f'section_image_url_{section_id}_{image_id}'] = image_urls[section_id][image_id].numpy()
                    data[f'section_image_captions_{section_id}_{image_id}'] = tf.sparse.to_dense(d[1]['section_image_captions'])[section_id][image_id].numpy()

            # page description task
            split = d[0]['split'].numpy().decode()
            is_sample = d[0]['is_page_description_sample'].numpy()
            if is_sample == 1:
                page_list[split].append(data)
            # section summarization task
            are_samples = tf.sparse.to_dense(d[1]['is_section_summarization_sample']).numpy()
            for section_id in range(are_samples.shape[0]):
                is_sample = are_samples[section_id][0]
                if is_sample == 1:
                    data['section_target'] = section_id
                    section_list[split].append(data)
            # image summarization task
            are_samples = tf.sparse.to_dense(d[1]['is_image_caption_sample']).numpy()
            for section_id in range(are_samples.shape[0]):
                for image_id in range(are_samples[section_id].shape[0]):
                    is_sample = are_samples[section_id][image_id]
                    if is_sample == 1:
                        data['section_image_target'] = (section_id, image_id)
                        image_list[split].append(data)

        print(f'task: page, train_num: ', len(page_list['train']), ', val_num: ', len(page_list['val']), ', test_num: ', len(page_list['test']))
        print(f'task: section, train_num: ', len(section_list['train']), ', val_num: ', len(section_list['val']), ', test_num: ', len(section_list['test']))
        print(f'task: image, train_num: ', len(image_list['train']), ', val_num: ', len(image_list['val']), ', test_num: ', len(image_list['test']))

        for split in ('train', 'val', 'test'):
            df = pd.DataFrame(page_list[split])
            df.to_parquet(f'{self.path}/page_{split}_small.parquet')
            df = pd.DataFrame(section_list[split])
            df.to_parquet(f'{self.path}/section_{split}_small.parquet')
            df = pd.DataFrame(image_list[split])
            df.to_parquet(f'{self.path}/image_{split}_small.parquet')

    def split_ids(self, task):
        id_list = defaultdict(list)
        for page_id, d in enumerate(self.dataset):
            if page_id % 100000 == 0:
                print(page_id, 'have processed...')
            if page_id == 600000:
                break
            if page_id < 400000:
                split = "train"
            elif page_id < 500000:
                split = "val"
            else:
                split = "test"
            #split = d[0]['split'].numpy().decode()
            if task == 'page':
                is_sample = d[0]['is_page_description_sample'].numpy()
                if is_sample == 0:
                    continue
                id_list[split].append(page_id)
            elif task == 'section':
                #are_samples = tf.sparse.to_dense(d[1]['is_section_summarization_sample']).numpy()
                are_samples = d[1]['is_section_summarization_sample'].values.numpy()
                for section_id in range(are_samples.shape[0]):
                    is_sample = are_samples[section_id]
                    if is_sample == 0:
                        continue
                    id_list[split].append((page_id, section_id))

        print(f'task: {task}, train_num: ', len(id_list['train']), ', val_num: ', len(id_list['val']), ', test_num: ', len(id_list['test']))
        with open(f'{self.path}/{task}_id_split_large.pkl', 'wb') as file:
            pickle.dump(id_list, file)


    def convert_to_numpy(self):
        numpy_list = []
        for d in self.dataset:
            page = {}
            page['split'] = d[0]['split'].numpy()
            page['page_title'] = d[0]['page_title'].numpy()
            page['page_url'] = d[0]['page_url'].numpy()
            page['clean_page_description'] = d[0]['clean_page_description'].numpy()
            page['raw_page_description'] = d[0]['raw_page_description'].numpy()
            page['is_page_description_sample'] = d[0]['is_page_description_sample'].numpy()
            page['page_contains_images'] = d[0]['page_contains_images'].numpy()
            page['page_content_sections_without_table_list'] = d[0]['page_content_sections_without_table_list'].numpy()

            section = {}
            section['is_section_summarization_sample'] = d[1]['is_section_summarization_sample'].values.numpy()
            section['section_title'] = d[1]['section_title'].values.numpy()
            section['section_index'] = d[1]['section_index'].values.numpy()
            section['section_depth'] = d[1]['section_depth'].values.numpy()
            section['section_heading_level'] = d[1]['section_heading_level'].values.numpy()
            section['section_subsection_index'] = d[1]['section_subsection_index'].values.numpy()
            section['section_parent_index'] = d[1]['section_parent_index'].values.numpy()
            section['section_text'] = d[1]['section_text'].values.numpy()
            section['section_clean_1st_sentence'] = d[1]['section_clean_1st_sentence'].values.numpy()
            section['section_raw_1st_sentence'] = d[1]['section_raw_1st_sentence'].values.numpy()
            section['section_rest_sentence'] = d[1]['section_rest_sentence'].values.numpy()
            section['is_image_caption_sample'] = d[1]['is_image_caption_sample'].values.numpy()
            section['section_image_url'] = d[1]['section_image_url'].values.numpy()
            section['section_image_mime_type'] = d[1]['section_image_mime_type'].values.numpy()
            section['section_image_width'] = d[1]['section_image_width'].values.numpy()
            section['section_image_height'] = d[1]['section_image_height'].values.numpy()
            section['section_image_in_wit'] = d[1]['section_image_in_wit'].values.numpy()
            section['section_contains_table_or_list'] = d[1]['section_contains_table_or_list'].values.numpy()
            section['section_image_captions'] = d[1]['section_image_captions'].values.numpy()
            section['section_image_alt_text'] = d[1]['section_image_alt_text'].values.numpy()
            section['section_image_raw_attr_desc'] = d[1]['section_image_raw_attr_desc'].values.numpy()
            section['section_image_clean_attr_desc'] = d[1]['section_image_clean_attr_desc'].values.numpy()
            section['section_image_raw_ref_desc'] = d[1]['section_image_raw_ref_desc'].values.numpy()
            section['section_image_clean_ref_desc'] = d[1]['section_image_clean_ref_desc'].values.numpy()
            section['section_contains_images'] = d[1]['section_contains_images'].values.numpy()

            numpy_list.append((page, section))

        with open(f'{self.path}/wikiweb2m.pkl', 'wb') as file:
            pickle.dump(numpy_list, file)

    def download_images(self):
        headers = {"User-Agent": "research (https://www.cs.cmu.edu/; minjiy@cs.cmu.edu)"}

        for page_id, d in enumerate(self.dataset):
            if page_id < 250000:
                continue
            if page_id == 300000:
                break
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
                    #file_name = f'{self.path}images/{page_id}_{section_id}_{image_id}.{file_format}'
                    file_name = f'/projects/rsalakhugroup/minjiy/images/{page_id}_{section_id}_{image_id}.{file_format}'
                    if os.path.exists(file_name):
                        break

                    another_image = False
                    try:
                        response = requests.get(image_url, headers=headers)
                        response.raise_for_status()
                    except requests.exceptions.HTTPError as e:
                        if "404 Client Error: Not Found for url" in str(e):
                            another_image = True
                            continue
                        else:
                            time.sleep(1)
                            response = requests.get(image_url)

                    with open(file_name, 'wb') as file:
                        for chunk in response.iter_content(8192):
                            file.write(chunk)

                    try:
                        img = Image.open(file_name)
                    except:
                        if os.path.exists(file_name):
                            os.remove(file_name)
                        another_image = True
                        continue

                    if another_image == False:
                        break


if __name__ == "__main__":
    parser = DataParser()
    #parser.convert_to_numpy()
    #parser.split_preprocess()
    #parser.split_ids('section')
    #parser.save_df_torch()
    parser.download_images()
