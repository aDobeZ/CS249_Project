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
    max_idx = -1
    for index in range(len(reader)):
        line = reader[index]
        str_temp = line.strip().split("\t")
        curr_index = int(str_temp[0].strip())
        node_index.append(curr_index)
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
        label_array[node_index[index]] = class_label_new
    return node_index, label_array, class_num

def parse_set_split_file(filename):
    np_indices = np.loadtxt(filename)
    """
    np_indices: numpy.array
    """
    indices = []
    for index in np_indices:
        indices.append(th.from_numpy(index.astype(int)))
    return indices

def process_cora(root_path):
    """ cora dataset process

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

    data_path = os.path.join(root_path, 'Cora')
    if not (os.path.exists(data_path)):
        print('Can not find cora in {}, please download the dataset first.'.format(data_path))

    #Construct graph from raw data.
    # paper_author
    paper_author_src, paper_author_dst = parse_edge_index_file(os.path.join(data_path, 'PA.txt'))

    # paper_term
    paper_term_src, paper_term_dst = parse_edge_index_file(os.path.join(data_path, 'PT.txt'))

    # paper_paper
    paper_paper_src, paper_paper_dst = parse_edge_index_file(os.path.join(data_path, 'PP.txt'))

    #build graph
    hg = dgl.heterograph({
        ('paper', 'pa', 'author') : (paper_author_src, paper_author_dst),
        ('author', 'ap', 'paper') : (paper_author_dst, paper_author_src),
        ('paper', 'pt', 'tag') : (paper_term_src, paper_term_dst),
        ('tag', 'tp', 'paper') : (paper_term_dst, paper_term_src),
        ('paper', 'pp1', 'paper') : (paper_paper_src, paper_paper_dst), 
        ('paper', 'pp2', 'paper') : (paper_paper_dst, paper_paper_src)})

    print("Graph constructed.")

    # Split data into train/eval/test
    all_y_index, all_y_label, class_num = \
            parse_label_index_file(os.path.join(data_path, 'paper_label.txt'))
    train_y_index = parse_set_split_file(os.path.join(data_path, 'paper_label_train_idx.txt'))
    test_y_index = parse_set_split_file(os.path.join(data_path, 'paper_label_test_idx.txt'))

    return hg, all_y_index, all_y_label, train_y_index, test_y_index, class_num
