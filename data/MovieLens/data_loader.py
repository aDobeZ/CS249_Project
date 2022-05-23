import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import os
import pickle as pkl
import numpy as np
import random

def parse_edge_index_file(filename):
    """Parse edge index file."""
    with open(filename, 'r') as file_to_read:
        reader = file_to_read.read().splitlines()
        src = []
        dst = []
    for index in range(3, len(reader)):
        line = reader[index]
        str_temp = line.strip().split("\t")
        row_index = int(str_temp[0].strip())
        col_index = int(str_temp[1].strip())
        src.append(row_index)
        dst.append(col_index)
    return src, dst


def parse_label_index_file(filename):
    """Parse label index file."""
    with open(filename, 'r') as file_to_read:
        reader = file_to_read.read().splitlines()
    node_index = []
    label = []
    label_list = []
    label_list_new = []
    count = 0
    for index in range(len(reader)):
        line = reader[index]
        str_temp = line.strip().split("\t")
        node_index.append(int(str_temp[0].strip()))
        label_temp = int(str_temp[1].strip())
        label.append(label_temp)
        if label_temp not in label_list:
            label_list.append(label_temp)
            label_list_new.append(count)
            count = count + 1
    class_num = len(label_list)
    label_array = np.zeros((len(label), class_num))
    for index in range(len(label)):
        class_label = label[index]
        idx = label_list.index(class_label)
        class_label_new = label_list_new[idx]
        label_array[index][class_label_new] = 1
    return node_index, label_array, class_num

# Split data into train/eval/test
def split_data(hg, etype_name):
    src, dst = hg.edges(etype=etype_name)
    user_item_src = src.numpy().tolist()
    user_item_dst = dst.numpy().tolist()
    
    num_link = len(user_item_src)
    pos_label=[1]*num_link
    pos_data=list(zip(user_item_src,user_item_dst,pos_label))

    ui_adj = np.array(hg.adj(etype=etype_name).to_dense())
    full_idx = np.where(ui_adj==0)

    sample = random.sample(range(0, len(full_idx[0])), num_link)
    neg_label = [0]*num_link
    neg_data = list(zip(full_idx[0][sample],full_idx[1][sample],neg_label))
    
    full_data = pos_data + neg_data
    random.shuffle(full_data)

    train_size = int(len(full_data) * 0.6)
    eval_size = int(len(full_data) * 0.2)
    test_size = len(full_data) - train_size - eval_size
    train_data = full_data[:train_size]
    eval_data = full_data[train_size : train_size+eval_size]
    test_data = full_data[train_size+eval_size : train_size+eval_size+test_size]
    train_data = np.array(train_data)
    eval_data = np.array(eval_data)
    test_data = np.array(test_data)
    
    return train_data, eval_data, test_data

def process_movielens(root_path):
    # User-Movie 943 1682 100000 UMUM
    # User-Age 943 8 943 UAUM
    # User-Occupation 943 21 943 UOUM
    # Movie-Genre 1682 18 2861 UMGM

    data_path = os.path.join(root_path, 'Movielens')
    if not (os.path.exists(data_path)):
        print('Can not find movielens in {}, please download the dataset first.'.format(data_path))

    #Construct graph from raw data.
    # movie_director
    movie_director_src, movie_director_dst = parse_edge_index_file(os.path.join(data_path, 'movie_director.txt'))

    # user_movie
    user_movie_src, user_movie_dst = parse_edge_index_file(os.path.join(data_path, 'user_movie_rating.txt'))

    # movie_tag
    movie_tag_src, movie_tag_dst = parse_edge_index_file(os.path.join(data_path, 'movie_tag.txt'))

    # movie_writer
    movie_writer_src, movie_writer_dst = parse_edge_index_file(os.path.join(data_path, 'movie_writer.txt'))

    #build graph
    hg = dgl.heterograph({
        ('movie', 'md', 'director') : (movie_director_src, movie_director_dst),
        ('director', 'dm', 'movie') : (movie_director_dst, movie_director_src),
        ('user', 'um', 'movie') : (user_movie_src, user_movie_dst),
        ('movie', 'mu', 'user') : (user_movie_dst, user_movie_src),
        ('movie', 'mt', 'tag') : (movie_tag_src, movie_tag_dst), 
        ('tag', 'tm', 'movie') : (movie_tag_dst, movie_tag_src),
        ('movie', 'ua', 'writer') : (movie_writer_src, movie_writer_dst),
        ('writer', 'au', 'movie') : (movie_writer_dst, movie_writer_src)})

    print("Graph constructed.")

    # Split data into train/eval/test
    train_data, eval_data, test_data = split_data(hg, 'um')

    return hg, train_data, eval_data, test_data
