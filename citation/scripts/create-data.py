#!/usr/bin/env python3

"""
Create the data for the citation experiments.
"""
import os
import random
import sys

import dgl
import numpy
import torch

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))
import util

DATASET_CITESEER = 'citeseer'
DATASET_CORA = 'cora'
DATASETS = [DATASET_CITESEER, DATASET_CORA]

DATASET_CONFIG = {
    DATASET_CITESEER: {
        "name": DATASET_CITESEER,
        "class-size": 6,
        "obs-size": 0,
        "train-size": 165,
        "valid-size": 165,
        "test-size": 1000,
        "num-splits": 10,
        "num-sgc-layers": 2,
    },
    DATASET_CORA: {
        "name": DATASET_CORA,
        "class-size": 7,
        "obs-size": 0,
        "train-size": 135,
        "valid-size": 135,
        "test-size": 1000,
        "num-splits": 10,
        "num-sgc-layers": 2,
    },
}

CONFIG_FILENAME = "config.json"


def _generate_partitions(graph, device, class_size, obs_count, train_count, test_count, valid_count):
    """
    Generate train, test, and valid partition masks. Guarantee at least one node per class for each partition.
    """
    found_sample = False
    while not found_sample:
        graph.ndata["obs-train-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["unobs-train-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["train-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["test-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["valid-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["latent-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)

        permutation = torch.randperm(graph.num_nodes(), device=device)

        graph.ndata["obs-train-mask"][permutation[:obs_count]] = True
        graph.ndata["unobs-train-mask"][permutation[obs_count: obs_count + train_count]] = True
        graph.ndata["train-mask"][permutation[:obs_count + train_count]] = True
        graph.ndata["valid-mask"][permutation[obs_count + train_count: obs_count + train_count + valid_count]] = True
        graph.ndata["test-mask"][permutation[obs_count + train_count + valid_count: obs_count + train_count + valid_count + test_count]] = True
        graph.ndata["latent-mask"][permutation[obs_count + train_count + valid_count + test_count:]] = True

        for mask_name in ["obs-train-mask", "unobs-train-mask", "train-mask", "test-mask", "valid-mask"]:
            found_sample = found_sample or len(torch.unique(graph.ndata["label"][graph.ndata[mask_name]])) == class_size

    return graph


def fetch_data(config):
    random.seed(config['seed'])
    numpy.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    dgl.seed(config['seed'])

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    if config['name'] == DATASET_CITESEER:
        graph = dgl.data.CiteseerGraphDataset()[0]
    elif config['name'] == DATASET_CORA:
        graph = dgl.data.CoraGraphDataset()[0]
    else:
        raise ValueError("Unknown dataset: '%s'." % (config['name'],))

    graph = dgl.add_self_loop(dgl.remove_self_loop(graph)).to(device)

    graph = _generate_partitions(graph, device, config['class-size'],
                                 config['obs-size'], config['train-size'],
                                 config['test-size'], config['valid-size'])

    # SGCN baseline.
    symmetric_graph = dgl.to_simple(dgl.add_reverse_edges(graph).to('cpu')).to(device)

    sgconv_layer = dgl.nn.pytorch.conv.SGConv(symmetric_graph.ndata['feat'].shape[1],
                                              symmetric_graph.ndata['feat'].shape[1],
                                              k=config['num-sgc-layers']).to(device)
    torch.nn.init.eye_(sgconv_layer.fc.weight)
    smoothed_features = sgconv_layer(symmetric_graph, symmetric_graph.ndata['feat']).detach().cpu().numpy()

    graph = dgl.remove_self_loop(graph)
    edges = graph.cpu().edges()
    edges = numpy.transpose(numpy.array([edges[0].numpy(), edges[1].numpy()]))

    data = {}
    for (partition, indexes) in (
            [('obs-train', graph.nodes()[graph.ndata["obs-train-mask"]]),
             ('unobs-train', graph.nodes()[graph.ndata["unobs-train-mask"]]),
             ('train', graph.nodes()[graph.ndata["train-mask"]]),
             ('test', graph.nodes()[graph.ndata["test-mask"]]),
             ('valid', graph.nodes()[graph.ndata["valid-mask"]]),
             ('latent', graph.nodes()[graph.ndata["latent-mask"]]),
             ('all', graph.nodes())]):
        data[partition] = {
            'entity-ids': [int(index) for index in indexes],
            'labels': [int(graph.ndata['label'][index]) for index in indexes],
            'features-simple': [graph.ndata['feat'][index].tolist() for index in indexes],
            'features-gcn': [graph.ndata['feat'][index].tolist() for index in indexes],
            'features-smoothed': [smoothed_features[index].tolist() for index in indexes],
        }

    return data, edges, graph


def write_data(config, out_dir, graph, data, edges):
    entity_data_map = []
    entity_data_smoothed_map = []

    for key in data:
        category_targets = []
        category_truth = []
        for entity_index in range(len(data[key]['entity-ids'])):
            entity = data[key]['entity-ids'][entity_index]

            entity_data_map.append([entity] + data[key]['features-simple'][entity_index] + [data[key]['labels'][entity_index]])
            entity_data_smoothed_map.append([entity] + data[key]['features-smoothed'][entity_index] + [data[key]['labels'][entity_index]])

            for label_index in range(config['class-size']):
                label = "0" if label_index != data[key]['labels'][entity_index] else "1"

                category_targets.append([entity, str(label_index)])
                category_truth.append([entity, str(label_index), label])

        util.write_psl_data_file(os.path.join(out_dir, "category-target-%s.txt" % key), category_targets)
        util.write_psl_data_file(os.path.join(out_dir, "category-truth-%s.txt" % key), category_truth)

    util.write_psl_data_file(os.path.join(out_dir, "edges.txt"), edges)

    util.write_psl_data_file(os.path.join(out_dir, "entity-data-map.txt"), entity_data_map)
    util.write_psl_data_file(os.path.join(out_dir, "entity-data-smoothed-map.txt"), entity_data_smoothed_map)
    util.write_json_file(os.path.join(out_dir, "deep-data.json"), data, indent=0)

    dgl.save_graphs(os.path.join(out_dir, 'dgl-graph.bin'), [graph])

    util.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def main():
    for dataset_id in DATASETS:
        config = DATASET_CONFIG[dataset_id]
        for split in range(config['num-splits']):
            config['seed'] = split

            out_dir = os.path.join(THIS_DIR, '..', 'data', 'experiment::' + dataset_id, 'split::' + str(split))
            os.makedirs(out_dir, exist_ok=True)

            if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                print("Data already exists for %s. Skipping generation." % out_dir)
                continue

            data, edges, graph = fetch_data(config)
            write_data(config, out_dir, graph, data, edges)


if __name__ == '__main__':
    main()
