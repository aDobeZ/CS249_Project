import dgl
import os
import numpy as np
import torch as th

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
    label_array = np.zeros((len(label), class_num), dtype=int)
    for index in range(len(label)):
        class_label = label[index]
        idx = label_list.index(class_label)
        class_label_new = label_list_new[idx]
        label_array[index][class_label_new] = 1
    return node_index, label_array, class_num

def parse_set_split_file(filename):
    np_indices = np.loadtxt(filename)
    """
    np_indices: numpy.array
    """
    indices = []
    for index in np_indices:
        indices.append(th.nonzero(th.from_numpy(index), as_tuple=False).squeeze())
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
    all_y_index, all_y_label, class_num = \
            parse_label_index_file(os.path.join(data_path, 'movie_genre.txt'))
    train_y_index = parse_set_split_file(os.path.join(data_path, 'movie_genre_train_idx.txt'))
    test_y_index = parse_set_split_file(os.path.join(data_path, 'movie_genre_test_idx.txt'))

    return hg, all_y_index, all_y_label, train_y_index, test_y_index
