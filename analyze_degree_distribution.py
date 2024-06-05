import argparse

import ipdb

# from data_loader import load_data
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import degree, to_undirected #, coalesce, remove_self_loops

import torch_geometric.transforms as T

from torch_geometric.typing import SparseTensor
import torch.nn.functional as F
import faiss

def create_index(features, use_gpu=False):
    # 定义索引类型，这里使用L2距离
    dimension = features.shape[1]  # 获取特征维度
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离

    if use_gpu:
        # 如果使用 GPU，将索引转移到 GPU
        gpu_res = faiss.StandardGpuResources()  # 使用标准 GPU 资源
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    index.add(features)  # 向索引中添加数据
    return index

def search_knn(index, queries, k):
    # 对查询向量进行 k-NN 搜索
    distances, indices = index.search(queries, k)
    return distances, indices


def normalize_L2(x):
    """归一化数组的每一行到单位L2范数"""
    faiss.normalize_L2(x)  # 直接使用faiss的归一化函数



def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    # return np.array(mask, dtype=np.bool)
    return np.array(mask, dtype=np.bool_)    # A6000 跑 paper100M 的时候用这行

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cora', help='See choices',
                        choices=['cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'ogbn-proteins', 'ogbn-products', 'papers100M'])
    args = parser.parse_args()
    dataset_str = args.dataset

    if dataset_str.startswith('ogb'):
        from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset


        dataset = PygNodePropPredDataset(dataset_str, root='./dataset')
        data = dataset[0]
        data0 = data.edge_index[0]
        data1 = data.edge_index[1]
        num_nodes = data.y.shape[0]
        out_degree_0 = degree(data0, num_nodes=num_nodes).long()  # 由edge_index第一行算出来的degree
        int_degree_1 = degree(data1, num_nodes=num_nodes).long()  # 由edge_index第二行算出来的degree
        all_edges = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)   # 无向图，考虑两个方向
        node_degree_0 = degree(all_edges[0], num_nodes=num_nodes).long()    # 转成无向图后，由edge_index第一行算出来的degree
        node_degree_1 = degree(all_edges[1], num_nodes=num_nodes).long()    # 转成无向图后，由edge_index第二行算出来的degree
        if torch.sum(node_degree_0 != node_degree_1) == 0:  # node_degree_0和node_degree_1是一样的
            print("out_degree_sym == in_degree_sym")
        else:
            print("error! node_degree_0 != node_degree_1")
            exit()
        if dataset_str == "ogbn-arxiv":     # arxiv数据集需要使用torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)
            node_degree = node_degree_0
        elif dataset_str == "ogbn-products":  # products 数据集不需要复制 edge_index
            node_degree = out_degree_0

        elif dataset_str == "ogbn-proteins":  # proteins 数据集不需要复制 edge_index
            node_degree = out_degree_0

        '''    # 以下片段跟上面的功能和结果一致
        dataset = PygNodePropPredDataset(dataset_str, root='./dataset', transform=T.ToSparseTensor())
        data2 = dataset[0]
        out_degree = data2.adj_t.sum(dim=1).long()     # 对于转制的矩阵 dim=1是原始矩阵的行和
        in_degree = data2.adj_t.sum(dim=0).long()
        data2.adj_t = data2.adj_t.to_symmetric()
        out_degree_sym = data2.adj_t.sum(dim=1).long()     # 对于转制的矩阵 dim=1是原始矩阵的行和
        in_degree_sym  = data2.adj_t.sum(dim=0).long()
        if torch.sum(out_degree_sym != in_degree_sym) == 0:  # node_degree_0和node_degree_1是一样的
            print("out_degree_sym == in_degree_sym")
        else:
            print("error! node_degree_0 != node_degree_1")
            exit()
        '''


        split_idx = dataset.get_idx_split()
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_mask = sample_mask(split_idx['train'], data.y.shape[0])
        val_mask = sample_mask(split_idx['valid'], data.y.shape[0])
        test_mask = sample_mask(split_idx['test'], data.y.shape[0])

        print(np.where(train_mask > 0)[0])
        print(np.where(train_mask > 0)[0].shape)
        print(np.where(val_mask > 0)[0])
        print(np.where(val_mask > 0)[0].shape)
        print(np.where(test_mask > 0)[0])
        print(np.where(test_mask > 0)[0].shape)



        train_node_degree = node_degree[train_mask]
        val_node_degree = node_degree[val_mask]
        test_node_degree = node_degree[test_mask]

        index_of_0_degree = torch.where(node_degree == 0)[0].numpy()        # degree=0的节点的索引
        index_of_train_mask = np.where(train_mask==True)[0]                 # train nodes的索引
        comment_index_with_train = np.intersect1d(index_of_0_degree, index_of_train_mask)   # degree=0 且是train nodes的索引
        index_of_0_degree_remove_train = np.setdiff1d(index_of_0_degree, index_of_train_mask)   # degree=0 去除train nodes且degree0的节点索引
        degre_0_X = data.x[index_of_0_degree_remove_train, :]   # 48377 * 100
        train_X = data.x[train_mask, :]                         # 196615 * 100
        # constLargeGraph(train_X, degre_0_X, k=5, step=5000)

        normalize_L2(train_X.numpy())
        normalize_L2(degre_0_X.numpy())
        index = create_index(train_X, use_gpu=False)  # 创建索引, 使用normalize_L2从而利用cosine距离
        k = 5  # 查找最近的 5 个邻居
        distances, indices = search_knn(index, degre_0_X, k)    # 执行搜索
        cosine_similarities = 1 - 0.5 * distances
        # index_of_train_mask = np.where(train_mask==True)[0] train_mask原始的索引是从0～196614
        # index_of_0_degree_remove_train degree为0的原始索引
        # indices [48377, 5]  每一行对应5个training samples的 index
        new_edges = torch.zeros((2, indices.shape[0] * indices.shape[1])).long()    # [2, 241885]
        new_edges[0] = torch.from_numpy(np.repeat(index_of_0_degree_remove_train, k))
        new_edges[1] = torch.from_numpy(indices.flatten())
        new_edges = to_undirected(new_edges)            # [2, 483770]
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
        data.edge_index = coalesce(data.edge_index, num_nodes=None, is_sorted=False)



    elif dataset_str == "papers100M" :          # papers100M 数据集使用A6000机器跑实验
        from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name='ogbn-papers100M', root='/ext/ogb_datasets')
        data = dataset[0]
        # ipdb.set_trace()

        data0 = data.edge_index[0]
        data1 = data.edge_index[1]
        num_nodes = data.y.shape[0]
        out_degree_0 = degree(data0, num_nodes=num_nodes)  # 由edge_index第一行算出来的degree
        int_degree_1 = degree(data1, num_nodes=num_nodes)  # 由edge_index第二行算出来的degree
        all_edges = torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)   # 无向图，考虑两个方向
        node_degree_0 = degree(all_edges[0], num_nodes=num_nodes).long()    # 转成无向图后，由edge_index第一行算出来的degree
        node_degree_1 = degree(all_edges[1], num_nodes=num_nodes).long()    # 转成无向图后，由edge_index第二行算出来的degree
        if torch.sum(node_degree_0 != node_degree_1) == 0:  # node_degree_0和node_degree_1是一样的
            print("out_degree_sym == in_degree_sym")
        else:
            print("error! node_degree_0 != node_degree_1")
            exit()
        # papers100M 数据集需要使用torch.cat([data.edge_index, data.edge_index[[1, 0]]], dim=1)
        node_degree = node_degree_0

        '''    # 以下片段跟上面的功能和结果一致
        dataset = PygNodePropPredDataset(dataset_str, root='./dataset', transform=T.ToSparseTensor())
        data2 = dataset[0]
        out_degree = data2.adj_t.sum(dim=1).long()     # 对于转制的矩阵 dim=1是原始矩阵的行和
        in_degree = data2.adj_t.sum(dim=0).long()
        data2.adj_t = data2.adj_t.to_symmetric()
        out_degree_sym = data2.adj_t.sum(dim=1).long()     # 对于转制的矩阵 dim=1是原始矩阵的行和
        in_degree_sym  = data2.adj_t.sum(dim=0).long()
        if torch.sum(out_degree_sym != in_degree_sym) == 0:  # node_degree_0和node_degree_1是一样的
            print("out_degree_sym == in_degree_sym")
        else:
            print("error! node_degree_0 != node_degree_1")
            exit()
        '''

        split_idx = dataset.get_idx_split()
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_mask = sample_mask(split_idx['train'], data.y.shape[0])
        val_mask = sample_mask(split_idx['valid'], data.y.shape[0])
        test_mask = sample_mask(split_idx['test'], data.y.shape[0])

        train_node_degree = node_degree[train_mask]
        val_node_degree = node_degree[val_mask]
        test_node_degree = node_degree[test_mask]


    else:
        dataset = Planetoid(name=dataset_str, root='./dataset')

        # dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
        # dataset = Planetoid(root='/tmp/Cora', name='Cora')
        # dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')

        # !!!!! Planetoid 中的三个数据集 load之后的edge本身就复制了一次，保证了是undirected的
        data = dataset[0]
        data0 = data.edge_index[0]
        data1 = data.edge_index[1]
        num_nodes = data.y.shape[0]
        out_degree_0 = degree(data0, num_nodes=num_nodes).long()  # 由edge_index第一行算出来的degree
        int_degree_1 = degree(data1, num_nodes=num_nodes).long()  # 由edge_index第二行算出来的degree
        print(f"difference between out_degree and in_degree: {torch.sum(out_degree_0 != int_degree_1)}")
        print(f" num of 0_degree nodes: {torch.where(out_degree_0 == 0)[0].shape}")
        print(f" num of 0_degree nodes: {torch.where(out_degree_0 == 1)[0].shape}")
        print(f" num of 0_degree nodes: {torch.where(out_degree_0 == 2)[0].shape}")
        adj = to_dense_adj(data.edge_index)[0]  # 默认得到一个[b, n, n]的矩阵， b为1 所以去掉。 也可以用 .squeeze(0)处理
        row_sum = torch.sum(adj, axis=0).long()
        col_sum = torch.sum(adj, axis=1).long()
        ## sum = row_sum + col_sum
        if torch.equal(row_sum, col_sum):
            print("Symmetric graph! row_sum == col_sum")
        else:
            print("Asymmetric graph! row_sum != col_sum")
        node_degree = row_sum
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask


        print(np.where(train_mask > 0)[0])
        print(np.where(train_mask > 0)[0].shape)
        print(np.where(val_mask > 0)[0])
        print(np.where(val_mask > 0)[0].shape)
        print(np.where(test_mask > 0)[0])
        print(np.where(test_mask > 0)[0].shape)


        #row_sum2 = load_data(args)
        #row_sum2 = torch.from_numpy(row_sum2)  # 和SLAPS 的codes得到的结果不一致 差三个数据 [3918],[6992],[8806]
        #features, nfeats, labels, nclasses, train_mask, val_mask, test_mask = load_data(args)
        train_node_degree = row_sum[train_mask]
        val_node_degree = row_sum[val_mask]
        test_node_degree = row_sum[test_mask]

        ## 找到 test node degree=0在原始数据集中的索引
        deg0_test_node_ind_in_test_mask = torch.nonzero(test_node_degree == 0).squeeze()  # degree=0的节点在test node 中的序列[ 47, 248, 261, 285, 408, 559, 564, 762, 802, 803, 870, 937]
        test_node_ind_in_allnodes = torch.nonzero(test_mask).squeeze()                # test_mask=True 在整个数据集的索引
        deg0_test_node_ind_in_allnodes = test_node_ind_in_allnodes[deg0_test_node_ind_in_test_mask] # degree=0的test node在整个数据集的索引 [2359, 2563, 2576, 2600, 2724, 2876, 2881, 3082, 3122, 3123, 3190, 3260]
        # test_mask[deg0_test_node_ind_in_allnodes]     # 验证这些索引是否存在于 test set中
        # node_degree[deg0_test_node_ind_in_allnodes]   # 验证这些索引是否degree=0

        ## 找到 test node degree=0在原始数据集中的索引 方法相同
        deg0_val_node_ind_in_test_mask = torch.nonzero(val_node_degree == 0).squeeze()
        val_node_ind_in_allnodes = torch.nonzero(val_mask).squeeze()
        deg0_val_node_ind_in_allnodes = val_node_ind_in_allnodes[deg0_val_node_ind_in_test_mask]
        # val_mask[deg0_val_node_ind_in_allnodes]     # 验证这些索引是否存在于 test set中
        # node_degree[deg0_val_node_ind_in_allnodes]   # 验证这些索引是否degree=0

        if dataset_str == "pubmed":
            remain_degree = node_degree.clone()     # 需要使用clone 以免更改原始的数据
            remain_degree[train_mask] = -1      # 去除 train set 的节点
            remain_degree[val_mask] = -1        # 去除 val set 的节点
            remain_degree[test_mask] = -1       # 去除 test set 的节点
            topk_v, topk_ins = torch.topk(remain_degree, torch.sum(train_mask))  # 取出之后再选前大60个值和索引
            if torch.sum(train_mask[topk_ins]) == 0 and torch.sum(val_mask[topk_ins]) == 0 and torch.sum(test_mask[topk_ins]) == 0:
                print("Top K degree nodes are not in training, val, test set!") # 确认topk不在测试或者验证集合中
                new_train_mask = torch.zeros(train_mask.shape[0], dtype=torch.bool)
                new_train_mask[topk_ins] = True
                if torch.sum(val_mask[torch.nonzero(new_train_mask)]) == 0 and torch.sum(test_mask[torch.nonzero(new_train_mask)]) == 0:
                    print("new_train_mask has no common nodes with val_mask and test_mask") # 确认train_mask和val_mask, test_mask没有公共节点
                new_train_node_degree = row_sum[new_train_mask]
                torch.save(new_train_mask, './new_mask/new_train_mask_pubmed.pt')
            else:
                print("Top K degree nodes are in labeled nodes. Rewrite the codes!!")
                exit()

    print(f"Node_degree: [{torch.min(node_degree)}, {torch.max(node_degree)}]")
    print(f"Train: [{torch.min(train_node_degree)}, {torch.max(train_node_degree)}]")
    print(f"Val: [{torch.min(val_node_degree)}, {torch.max(val_node_degree)}]")
    print(f"Test: [{torch.min(test_node_degree)}, {torch.max(test_node_degree)}]")
    # torch.nonzero(test_node_degree == 0).squeeze() # 在1000中，找到degree为0 的节点索引, 该索引不对应原始所有节点的索引
    # torch.sum(node_degree==0) # 统计degree=0的节点个数
    # torch.nonzero(node_degree == 0).squeeze() # 找到degree为0 的节点索引， 该索引对应原始所有节点的索引
    ''' # 以下代码确认degree为0 的节点是不是在 all_edges中不存在对应的边
    zero_index = torch.nonzero(node_degree == 0).squeeze()
    for value in zero_index:
        rows, cols = torch.where(all_edges == value)
        if rows.numel() == 0 or cols.numel() == 0:
            print("Not found!")
        else:
            print(f"Value {value} found at indices:")
            for row, col in zip(rows, cols):
                print(f"Row: {row.item()}, Column: {col.item()}")
    '''


    from collections import Counter

    occurrences = Counter(node_degree.numpy())
    sorted_occurrences = dict(sorted(occurrences.items()))
    x_values = np.array(list(sorted_occurrences.keys()))
    y_values = np.array(list(sorted_occurrences.values()))


    # train node degree distribution
    occurrences_train = Counter(train_node_degree.numpy())
    sorted_occurrences_train = dict(sorted(occurrences_train.items()))
    x_values_train = np.array(list(sorted_occurrences_train.keys()))
    y_values_train = np.array(list(sorted_occurrences_train.values()))


    # val node degree distribution
    occurrences_val = Counter(val_node_degree.numpy())
    sorted_occurrences_val = dict(sorted(occurrences_val.items()))
    x_values_val = np.array(list(sorted_occurrences_val.keys()))
    y_values_val = np.array(list(sorted_occurrences_val.values()))


    # test node degree distribution
    occurrences_test = Counter(test_node_degree.numpy())
    sorted_occurrences_test = dict(sorted(occurrences_test.items()))
    x_values_test = np.array(list(sorted_occurrences_test.keys()))
    y_values_test = np.array(list(sorted_occurrences_test.values()))


    '''
    occurrences_train_new = Counter(new_train_node_degree.numpy())
    sorted_occurrences_train_new = dict(sorted(occurrences_train_new.items()))
    x_values_train_new = np.array(list(sorted_occurrences_train_new.keys()))
    y_values_train_new = np.array(list(sorted_occurrences_train_new.values()))
    '''


    '''  '''  # train node degree distribution
    plt.figure(figsize=(8, 6))
    # plt.plot(x_values, y_values, color='blue', label='Curve of In-Degree Distribution')   # 每个顶点连成曲线
    plt.bar(x_values, y_values, color='gray', label='All Nodes')
    plt.bar(x_values_train, y_values_train, color='red', label='Train Nodes')
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Number of degrees (d)', fontdict={'size': 18})
    plt.ylabel('Number of nodes with d degrees', fontdict={'size': 18})
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.legend(prop={'size': 18})
    plt.savefig(f'figs/{dataset_str}_ori_train_and_all_degree.png')
    # plt.savefig(f'figs/{dataset_str}_ori_train_and_all_degree_xlog.png')
    plt.show()

    '''  '''  # val node degree distribution
    plt.figure(figsize=(8, 6))
    plt.bar(x_values, y_values, color='gray', label='All Nodes')
    plt.bar(x_values_val, y_values_val, color='blue', label='Val Nodes')
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Number of degrees (d)', fontdict={'size': 18})
    plt.ylabel('Number of nodes with d degrees', fontdict={'size': 18})
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.legend(prop={'size': 18})
    plt.savefig(f'figs/{dataset_str}_ori_val_and_all_degree.png')
    # plt.savefig(f'figs/{dataset_str}_ori_val_and_all_degree_xlog.png')
    plt.show()


    '''  '''  # val node degree distribution
    plt.figure(figsize=(8, 6))
    plt.bar(x_values, y_values, color='gray', label='All Nodes')
    plt.bar(x_values_test, y_values_test, color='green', label='Test Nodes')
    #plt.bar(x_values_train_new, y_values_train_new, color='green', label='New Labeled Nodes')
    plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Number of degrees (d)', fontdict={'size': 18})
    plt.ylabel('Number of nodes with d degrees', fontdict={'size': 18})
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.legend(prop={'size': 18})
    plt.savefig(f'figs/{dataset_str}_ori_test_and_all_degree.png')
    # plt.savefig(f'figs/{dataset_str}_ori_test_and_all_degree_xlog.png')
    plt.show()
