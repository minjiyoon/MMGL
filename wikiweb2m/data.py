import os
import time
import torch
from transformers import AutoTokenizer
import pickle
import pandas as pd
from PIL import Image
from urllib.request import urlopen

from language_modelling import utils
from torch_geometric.data import Data

def load_wikiweb2m(task):
    """
    Load WikiWeb2M dataset
    Args:
        task: 'section' for section summarization task
    Returns:
        train_df: pandas dataframe for training_set
        val_df: pandas dataframe for validation_set
        test_df: pandas dataframe for test_set
        id_list: list of (page_id, section_id) pairs
    """
    train_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_train_large.parquet')
    val_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_val_large.parquet')
    test_df = pd.read_parquet(f'./wikiweb2m/raw/wikiweb2m_test_large.parquet')

    with open(f'./wikiweb2m/raw/{task}_id_split_large.pkl', 'rb') as f:
        id_list = pickle.load(f)

    return train_df, val_df, test_df, id_list


class WikiWeb2M(torch.utils.data.Dataset):
    """
    WikiWeb2M dataset
    Args:
        args: args
        df: pandas dataframe for dataset
        id_list: list of (page_id, section_id) pairs
        tokenizer: tokenizer
        visual_feature_extractor_model: visual feature extractor model
    """

    def __init__(self, args, df, id_list, tokenizer, visual_feature_extractor_model=None):
        self.path = './wikiweb2m/raw/'
        self.image_path = '/projects/rsalakhugroup/minjiy/images/'

        self.task = args.task
        self.context = args.context
        self.decoder_only = args.decoder_only
        self.neighbor_mode = args.neighbor_mode

        self.max_text_neighbors = args.max_text_neighbors
        self.max_image_neighbors = args.max_image_neighbors
        self.position_type = args.position_type

        self.df = df
        self.id_list = id_list
        self.tokenizer = tokenizer
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length

        if visual_feature_extractor_model is not None and self.context in ('section_all', 'all'):
            self.visual_feature_extractor = utils.get_feature_extractor_for_model(visual_feature_extractor_model)

        self.n_text_tokens = args.n_text_tokens
        self.n_visual_tokens = args.n_visual_tokens

    def __len__(self):
        """
        Get dataset length
        Returns:
            dataset length
        """
        return len(self.id_list)

    def get_page_info(self, d):
        """
        Get page text information
        Args:
            d: pandas dataframe for dataset
        Returns:
            page_info: page information in raw text   
        """
        page_url = d['page_url'].decode()
        page_title = d['page_title'].decode()
        page_description = d['page_description'].decode()
        page_info = ', '.join([page_title, page_description])
        return ' '.join(page_info.replace('\n', ' ').split())

    def get_section_info(self, section_id, d, remove_summary=True):
        """
        Get section text information
        Args:
            section_id: section id
            d: pandas dataframe for dataset
            remove_summary: whether to remove summary (ground truth for section summarization) or not
        Returns:
            section_info: section information in raw text
            section_summary: section summary in raw text
        """
        section_depth = str(d['section_depth'][section_id])
        section_heading = str(d['section_heading'][section_id])
        section_parent_index = str(d['section_parent_index'][section_id])
        section_title = d['section_title'][section_id].decode()
        section_summary = d['section_summary'][section_id].decode()
        section_rest_sentence = d['section_rest_sentence'][section_id].decode()
        if remove_summary:
            section_info = ', '.join([section_rest_sentence])
            section_info, section_summary = ' '.join(section_info.replace('\n', ' ').split()), ' '.join(section_summary.replace('\n', ' ').split())
            return section_info, section_summary
        else:
            section_info = ', '.join([section_summary, section_rest_sentence])
            section_info = ' '.join(section_info.replace('\n', ' ').split())
            return section_info

    def get_section_images(self, page_id, section_id, d):
        """
        Get section image information
        Args:
            page_id: page id
            section_id: section id
            d: pandas dataframe for dataset
        Returns:
            section_image: section image
            section_caption: section image caption
        """
        section_num = d['section_title'].shape[0]
        image_urls = d['image_url'].reshape(section_num, -1)
        image_captions = d['image_caption'].reshape(section_num, -1)
        for image_id in range(image_urls[section_id].shape[0]):
            image_url = image_urls[section_id][image_id].decode()
            file_format = os.path.splitext(image_url)[1][1:]
            file_name = f'{self.image_path}/{page_id}_{section_id}_{image_id}.{file_format}'
            if os.path.exists(file_name):
                try:
                    img = Image.open(f'./wikiweb2m/raw/images/{page_id}_{section_id}_{image_id}.{file_format}')
                    section_image = utils.get_pixel_values_for_model(self.visual_feature_extractor, img)
                    section_caption = image_captions[section_id][image_id].decode()
                    return section_image, ' '.join(section_caption.replace('\n', ' ').split())
                except:
                    continue
        return None, None

    def __getitem__(self, index):
        """
        Get item
        Args:
            index: index
        Returns:
            dictionary of {
                input_ids: tokenized input ids,
                attention_mask: attention mask for input ids,
                labels: tokenized label ids,
                images: image features (only for section_all and all),
                image_positions: image locations among texts (only for section_all and all),
                neighbor_input_ids: tokenized input ids for neighbor texts,
                neighbor_attention_mask: attention mask for neighbor texts,
                neighbor_pos_ids: position ids for neighbor texts,
                text_locations: neighbor text embedding locations among embeddings,
                neighbor_images: image features for neighbor images,
                neighbor_images_pos_ids: position ids for neighbor images,
                image_locations: neighbor image embedding locations among embeddings
            }
        """
        if self.neighbor_mode == "embedding":
            return self.get_embedding_item(index)

        page_id, section_id = self.id_list[index]
        d = self.df[self.df['page_id'] == page_id].iloc[0]
        if self.context == 'section_only':
            # Get section text information only
            section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
            inputs = 'summarize: ' + section_info
            input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]

        elif self.context == "section_all":
            # Get section text and image information
            section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
            image, image_caption = self.get_section_images(page_id, section_id, d)

            images = []
            image_positions = []
            if image is None:
                # No image in the section
                inputs = "summarize: " + section_info
                visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                images.append(torch.zeros((3,  224, 224)))
            else:
                # If image exists, add image caption to the input text
                inputs = "summarize: " + section_info + ", conext: " + image_caption
                visual_ids = torch.LongTensor(self.n_visual_tokens * [-1])
                images.append(image)
            max_text_length = self.max_input_length - self.n_visual_tokens
            input_ids = self.tokenizer(inputs, max_length=max_text_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            # Image is concatenated at the end of the input text
            image_positions.append(input_ids.shape[0] + torch.arange(self.n_visual_tokens))
            input_ids = torch.cat([input_ids, visual_ids], dim=0)

        elif self.context == "text_only":
            # Get all text information in the page
            page_info = self.get_page_info(d)
            section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
            # Collect text information from other sections
            context_info = []
            for context_id in range(len(d['section_title'])):
                if context_id == section_id:
                    continue
                context_info.append(self.get_section_info(context_id, d, remove_summary=False))
            context_info = ', '.join(context_info)
            inputs = "summarize: " + section_info + ", context: " + page_info + context_info
            input_ids = self.tokenizer(inputs, max_length=self.max_input_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]

        elif self.context == "all":
            # Get all text and image information in the page
            page_info = self.get_page_info(d)
            # Get text and image information in the target section
            section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
            section_image, section_caption = self.get_section_images(page_id, section_id, d)

            images = []
            image_positions = []
            if section_image is None:
                # No image in the section
                inputs = "summarize: " + section_info
                visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                images.append(torch.zeros((3,  224, 224)))
            else:
                # If image exists, add image caption to the input text
                inputs = "summarize: " + section_info + ", conext: " + section_caption
                visual_ids = torch.LongTensor(self.n_visual_tokens * [-1])
                images.append(section_image)
            max_text_length = self.max_input_length - self.n_visual_tokens
            input_ids = self.tokenizer(inputs, max_length=max_text_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            # Image is concatenated at the end of the input text from the target section
            image_positions.append(input_ids.shape[0] + torch.arange(self.n_visual_tokens))
            input_ids = torch.cat([input_ids, visual_ids], dim=0)

            # Collect text and image information from other sections
            for context_id in range(len(d['section_title'])):
                if context_id == section_id:
                    continue
                context_info = self.get_section_info(context_id, d, remove_summary=False)
                context_image, context_caption = self.get_section_images(page_id, context_id, d)
                if context_image is None:
                    # No image in the section
                    context = context_info
                    visual_ids = torch.LongTensor(self.n_visual_tokens * [self.tokenizer.pad_token_id])
                    context_image = torch.zeros((3,  224, 224))
                else:
                    # If image exists, add image caption to the input text
                    context = context_info + context_caption
                    visual_ids = torch.LongTensor(self.n_visual_tokens * [-1])
                max_text_length = self.max_input_length - input_ids.shape[0] - self.n_visual_tokens
                context_ids = self.tokenizer(context, max_length=max_text_length, padding="do_not_pad", truncation=False, return_tensors="pt").input_ids[0]
                if input_ids.shape[0] + context_ids.shape[0] + visual_ids.shape[0] > self.max_input_length:
                    break
                images.append(context_image)
                # Image is concatenated at the end of the input text from the corresponding section
                image_positions.append(input_ids.shape[0] + context_ids.shape[0] + torch.arange(self.n_visual_tokens))
                input_ids = torch.cat([input_ids, context_ids, visual_ids], dim=0)

            if len(input_ids) > self.max_input_length:
                input_ids = input_ids[:self.max_input_length]

        if self.decoder_only:
            # For decode-only models, labels are the same as inputs
            model_inputs = self.tokenizer.pad({"input_ids": [input_ids]}, max_length=self.max_input_length, padding="max_length", return_tensors="pt")
            labels = ", summary: " + labels
            label_ids = self.tokenizer(labels, max_length=self.max_output_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            # Remove SOS token and add EOS token
            label_ids = torch.cat([label_ids[1:], torch.LongTensor([self.tokenizer.eos_token_id])], dim=0)
            model_outputs = self.tokenizer.pad({"input_ids": [label_ids]}, max_length=self.max_output_length, padding="max_length", return_tensors="pt")

            result = {"input_ids": torch.cat((model_inputs.input_ids[0], model_outputs.input_ids[0]), dim=0),\
                      "attention_mask": torch.cat((model_inputs.attention_mask[0], model_outputs.attention_mask[0]), dim=0),\
                      "labels": torch.cat((model_inputs.input_ids[0], model_outputs.input_ids[0]), dim=0)}

        else:
            # For encoder-decoder models, labels are the summary of the target section
            model_inputs = self.tokenizer.pad({"input_ids": [input_ids]}, max_length=self.max_input_length, padding="max_length", return_tensors="pt")
            labels = self.tokenizer(labels, max_length=self.max_output_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
            labels_with_ignore_index = torch.LongTensor([label if label != 0 else -100 for label in labels])
            result = {"input_ids": model_inputs.input_ids[0], "attention_mask": model_inputs.attention_mask[0], "labels": labels_with_ignore_index}

        if self.context in ("section_all", "all"):
            # When image information is included, add image features and image locations
            images = torch.stack(images, dim=0)
            image_positions = torch.cat(image_positions, dim=0)
            result["images"] = images
            result["image_positions"] = image_positions

        return result

    def get_embedding_item(self, index):
        """
        Get item for embedding mode
        Args:
            index: index
        Returns:
            dictionary of {
                input_ids: tokenized input ids,
                attention_mask: attention mask for input ids,
                labels: tokenized label ids,
                neighbor_input_ids: tokenized input ids for neighbor texts,
                neighbor_attention_mask: attention mask for neighbor texts,
                neighbor_pos_ids: position ids for neighbor texts,
                text_locations: neighbor text embedding locations among embeddings,
                neighbor_images: image features for neighbor images,
                neighbor_images_pos_ids: position ids for neighbor images,
                image_locations: neighbor image embedding locations among embeddings
            }   
        """
        page_id, section_id = self.id_list[index]
        d = self.df[self.df['page_id'] == page_id].iloc[0]

        # Get section text information
        section_info, labels = self.get_section_info(section_id, d, remove_summary=True)
        inputs = "summarize: " + section_info
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, padding="max_length", truncation=True, return_tensors="pt")

        if self.decoder_only:
            # For decode-only models, labels are the same as inputs
            labels = ", summary: " + labels
            label_ids = self.tokenizer(labels, max_length=self.max_output_length, padding="do_not_pad", truncation=True, return_tensors="pt").input_ids[0]
            # Remove SOS token and add EOS token
            label_ids = torch.cat([label_ids[1:], torch.LongTensor([self.tokenizer.eos_token_id])], dim=0)
            model_outputs = self.tokenizer.pad({"input_ids": [label_ids]}, max_length=self.max_output_length, padding="max_length", return_tensors="pt")

            result = {"input_ids": torch.cat((model_inputs.input_ids[0], model_outputs.input_ids[0]), dim=0), \
                    "attention_mask": torch.cat((model_inputs.attention_mask[0], model_outputs.attention_mask[0]), dim=0), \
                    "labels": torch.cat((model_inputs.input_ids[0], model_outputs.input_ids[0]), dim=0)}
        else:
            # For encoder-decoder models, labels are the summary of the target section
            labels = self.tokenizer(labels, max_length=self.max_output_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
            labels_with_ignore_index = torch.LongTensor([label if label != 0 else -100 for label in labels])
            result = {"input_ids": model_inputs.input_ids[0], "attention_mask": model_inputs.attention_mask[0], "labels": labels_with_ignore_index}

        # Raw text and image features for neighbor texts and images
        neighbor_texts = []
        neighbor_images = []
        # Position ids for position embeddings
        position_texts = []
        position_images = []
        # Locations among neighbor embeddings
        location_texts = []
        location_images = []
        location = 0
        # Graph
        graph_index = {section_id: 0} # input text: 0, neighbors: location + 1
        edge_list = []

        #(1) page information
        page_info = self.get_page_info(d)
        neighbor_texts.append(page_info)
        position_texts.append(len(position_texts))
        location_texts.append(location)
        location += 1
        # Graph: input_text <-> page description
        edge_list.append((graph_index[section_id], location))

        #(2) section image information
        section_image, section_caption = self.get_section_images(page_id, section_id, d)
        if section_image is not None:
            neighbor_images.append(section_image)
            position_images.append(len(position_images))
            location_images.append(location)
            location += 1
            # Graph: input_text <-> image
            edge_list.append((graph_index[section_id], location))
            previous_image_id = location
            
            neighbor_texts.append(section_caption)
            position_texts.append(len(position_texts))
            location_texts.append(location)
            location += 1
            # Graph: input_text <-> caption
            edge_list.append((graph_index[section_id], location))
            # Graph: image <-> caption
            edge_list.append((previous_image_id, location))

        #(3) other section information from the same page
        previous_section_id = -1
        for context_id in range(len(d['section_title'])):
            if context_id == section_id:
                continue

            if len(neighbor_texts) < self.max_text_neighbors:
                context_info = self.get_section_info(context_id, d, remove_summary=False)
                neighbor_texts.append(context_info)
                position_texts.append(len(position_texts))
                location_texts.append(location)
                location += 1
                # Graph: previous section - current section (order)
                if previous_section_id > -1:
                    edge_list.append((previous_section_id, location))
                graph_index[context_id] = location
                previous_section_id = location
                
            if len(neighbor_images) < self.max_image_neighbors:
                context_image, context_caption = self.get_section_images(page_id, context_id, d)
                if context_image is not None:
                    neighbor_images.append(context_image)
                    position_images.append(len(position_images))
                    location_images.append(location)
                    location += 1
                    # Graph: section <-> image
                    edge_list.append((previous_section_id, location))
                    previous_image_id = location

                    if len(neighbor_texts) < self.max_text_neighbors:
                        neighbor_texts.append(context_caption)
                        position_texts.append(len(position_texts))
                        location_texts.append(location)
                        location += 1
                        # Graph: section <-> caption
                        edge_list.append((previous_section_id, location))
                        # Graph: image <-> caption
                        edge_list.append((previous_image_id, location))
        
        # Graph: hierachical relations
        for context_id in range(len(d['section_parent_index'])):
            parent_id = d['section_parent_index'][context_id]
            if context_id in graph_index.keys() and parent_id in graph_index.keys():
                edge_list.append((graph_index[context_id], graph_index[parent_id]))

        # PyG graph data
        node_num = 1 + self.max_text_neighbors + self.max_image_neighbors
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        if self.position_type == 'laplacian':
            node_value = torch.zeros((node_num))
            graph = Data(x=node_value, edge_index=edge_index)
            lpe = utils.compute_LPE(graph)
        elif self.position_type == 'gnn':
            edge_value = torch.ones((edge_index.shape[1]))
            graph = torch.sparse_coo_tensor(edge_index, edge_value, [node_num, node_num]).to_dense()
            graph = utils.normalize_graph(graph)
        
        # Increase position ids by 1 for padding_id
        position_texts = [position_id + 1 for position_id in position_texts]
        position_images = [position_id + 1 for position_id in position_images]
        # Pad texts
        while len(neighbor_texts) < self.max_text_neighbors:
            neighbor_texts.append('')
            position_texts.append(0)
            location_texts.append(location)
            location += 1
        # Pad images
        while len(neighbor_images) < self.max_image_neighbors:
            neighbor_images.append(torch.zeros((3,  224, 224)))
            position_images.append(0)
            location_images.append(location)
            location += 1

        #Tokenize neighbor texts
        neighbor_texts = self.tokenizer(neighbor_texts, max_length=self.max_input_length, padding="max_length", truncation=True, return_tensors="pt")
        result["neighbor_input_ids"] = neighbor_texts.input_ids
        result["neighbor_attention_mask"] = neighbor_texts.attention_mask
        result["neighbor_pos_ids"] = torch.LongTensor(position_texts)
        result["text_locations"] = torch.LongTensor(location_texts)
        result["neighbor_images"] = torch.stack(neighbor_images, dim=0)
        result["neighbor_images_pos_ids"] = torch.LongTensor(position_images)
        result["image_locations"] = torch.LongTensor(location_images)
        if self.position_type == 'laplacian':
            result["lpe"] = lpe
        if self.position_type == 'gnn':
            result["graph"] = graph
        return result


