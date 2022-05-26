import dgl
import os
import numpy as np
import torch as th

index_min = {}

def parse_edge_index_file(src_name, dst_name, filename):
    """Parse edge index file."""
    with open(filename, 'r') as file_to_read:
        reader = file_to_read.read().splitlines()
        src = []
        dst = []
    line = reader[1]
    str_temp = line.strip().split("\t")
    if src_name not in index_min:
        index_min[src_name] = int(str_temp[1].strip())
    if dst_name not in index_min:
        index_min[dst_name] = int(str_temp[2].strip())
    src_min = index_min[src_name]
    dst_min = index_min[dst_name]
    for index in range(3, len(reader)):
        line = reader[index]
        str_temp = line.strip().split("\t")
        row_index = int(str_temp[0].strip()) - src_min
        col_index = int(str_temp[1].strip()) - dst_min
        src.append(row_index)
        dst.append(col_index)
    return src, dst


def parse_label_index_file(name, filename):
    """Parse label index file."""
    with open(filename, 'r') as file_to_read:
        reader = file_to_read.read().splitlines()
    origin_index = []
    new_index = []
    label = []
    label_list = []
    label_list_new = []
    count = 0
    max_idx = -1
    for index in range(len(reader)):
        line = reader[index]
        str_temp = line.strip().split("\t")
        curr_index = int(str_temp[0].strip())
        origin_index.append(curr_index)
        new_index.append(curr_index - index_min[name])
        max_idx = max(max_idx, curr_index)
        label_temp = int(str_temp[1].strip())
        label.append(label_temp)
        if label_temp not in label_list:
            label_list.append(label_temp)
            label_list_new.append(count)
            count = count + 1
    class_num = len(label_list)
    label_array = np.empty((max_idx + 1), dtype=int)
    for index in range(len(label)):
        class_label = label[index]
        idx = label_list.index(class_label)
        class_label_new = label_list_new[idx]
        label_array[new_index[index]] = class_label_new
    return origin_index, new_index, label_array, class_num

def parse_set_split_file(name, filename):
    np_indices = np.loadtxt(filename)
    """
    np_indices: numpy.array
    """
    indices = []
    for index in np_indices:
        temp = np.ones(index.shape) * index_min[name]
        new_index = index - temp
        indices.append(th.from_numpy(new_index.astype(int)))
    return indices

def process_movielens(root_path):
    # User-Movie 943 1682 100000 UMUM
    # User-Age 943 8 943 UAUM
    # User-Occupation 943 21 943 UOUM
    # Movie-Genre 1682 18 2861 UMGM
    """ movielens dataset process

        Parameters
        ----------
        root_path : string
            The root of data folder

        Returns
        -------
        g : DGLHeteroGraph
            Heterogenoues graph. Same as g in entity_classify.py
        all_y_index: list
            a list of node index of labeled nodes.
        all_y_label: np.array
            stores a label matrix. One hot vector.
        train_y_index: list[torch.Tensor]
            A list of train_index in tensor format. Each tensor acts like train_idx in entity_classify.py
        test_y_index: list[torch.Tensor]
            A list of test_index in tensor format. Each tensor acts like test_index in entity_classify.py
        """

    data_path = os.path.join(root_path, 'MovieLens')
    if not (os.path.exists(data_path)):
        print('Can not find movielens in {}, please download the dataset first.'.format(data_path))

    #Construct graph from raw data.
    # movie_director
    movie_director_src, movie_director_dst = parse_edge_index_file('movie', 'director', os.path.join(data_path, 'movie_director.txt'))

    # user_movie
    user_movie_src, user_movie_dst = parse_edge_index_file('user', 'movie', os.path.join(data_path, 'user_movie_rating.txt'))

    # movie_tag
    movie_tag_src, movie_tag_dst = parse_edge_index_file('movie', 'tag', os.path.join(data_path, 'movie_tag.txt'))

    # movie_writer
    movie_writer_src, movie_writer_dst = parse_edge_index_file('movie', 'writer', os.path.join(data_path, 'movie_writer.txt'))

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
    all_y_index, all_y_new_index, all_y_label, class_num = \
            parse_label_index_file('movie', os.path.join(data_path, 'movie_genre.txt'))
    train_y_index = parse_set_split_file('movie', os.path.join(data_path, 'movie_genre_train_idx.txt'))
    test_y_index = parse_set_split_file('movie', os.path.join(data_path, 'movie_genre_test_idx.txt'))

    return hg, all_y_index, all_y_label, train_y_index, test_y_index, class_num
