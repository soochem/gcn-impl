# import tarfile
# import urllib.request
# import progressbar
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from os import listdir
from os.path import isfile, join
# import logging


def load_data_from_file(dir_path, dataset_name='cora'):
    """
    Parse txt file to generate dataset
    Reference
    :param dir_path: directory name where files exist
    :param dataset_name: dataset name from file name
    :return:
    """
    # Get all data file list under dir_path
    # data_files = [join(dir_path, f) for f in listdir(dir_path)
    #               if isfile(join(dir_path, f))]
    # print("Data file list : ", str(data_files))
    content_path = join(dir_path, dataset_name + '.content')
    cite_path = join(dir_path, dataset_name + '.cites')

    # if not path.exists(output_path):
    #     makedirs(output_path)

    # .content: < paper_id > < word_attributes > + < class_label >
    ids_features_labels = np.genfromtxt(content_path,
                                        dtype=np.dtype(str))
    ids = np.array(ids_features_labels[:, 0], dtype=np.int32)
    features = np.array(ids_features_labels[:, 1:-1], dtype=np.float32)
    features = torch.tensor(features)
    labels = OneHotEncoder(sparse=False).fit_transform(ids_features_labels[:, -1].reshape(-1,1))

    # .cites: <ID of cited paper> <ID of citing paper>
    edges_unordered = np.genfromtxt(cite_path,
                                    dtype=np.int32)

    # Build graph
    # edges : ((j:i) == (ids:i) => [35	1033 35	1032 ...]
    # [[ 163  402] => idx_map[edges_unordered[0]] idx_map[edges_unordered[1]]
    #  [ 163  659]
    idx_map = {j: i for i, j in enumerate(ids)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # adjacent matrix
    # S = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0, 1, 2], [2, 3, 0, 3]]), values=torch.tensor([1, 2, 1, 3]),
    #                             size=[3, 4])
    # # indices has x and y values separately along the 2 rows
    adj = torch.sparse_coo_tensor(indices=torch.tensor([edges[:, 0], edges[:, 1]]),
                                  values=torch.tensor(np.ones(edges.shape[0])),
                                  size=[labels.shape[0], labels.shape[0]])

    # Build symmetric adjacency matrix
    adj = adj + adj.transpose(0, 1)  # (2708, 2708), square matrix

    # Normalize
    features = normalize_data(features)  # input : ndarray
    adj = normalize_data(torch.eye(adj.shape[0]) + adj)  # input : torch.sparse.tensor, 더하는 순서가 상관있음?

    # Data Type ?
    features = torch.FloatTensor(features)
    # 처음 labels 를 써도 되지 않을까 - X
    # 다음과 동일 - LabelEncoder().fit_transform(ids_features_labels[:, -1])
    labels = torch.LongTensor(np.where(labels)[1])

    return adj, features, labels


def normalize_data(x):
    """
    Row-normalize sparse matrix
    """
    # sum - axis = 0: col, 1: row
    row_sum = np.array(x.sum(1))
    # inverse of row sum
    row_inv = np.power(row_sum, -1).flatten()
    row_inv[np.isinf(row_inv)] = 0.
    x = np.diag(row_inv).dot(x)
    return x


# ------- test ------- #
DATA_PATH = '../data/cora/'

# # Parse dataset from xml files
load_data_from_file(DATA_PATH, dataset_name='cora')  # cora_test => encoding 시 데이터가 못자라서 맵핑이 안되는 듯
