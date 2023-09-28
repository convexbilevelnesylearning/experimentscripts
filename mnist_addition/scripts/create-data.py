#!/usr/bin/env python3

# Construct the data and neural model for this experiment.
# Before a directory is generated, the existence of a config file for that directory will be checked,
# if it exists generation is skipped.

import os
import sys

import numpy
import torchvision

from itertools import product

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))
import util

DATASET_MNIST_1 = 'mnist-1'
DATASET_MNIST_2 = 'mnist-2'
DATASETS = [DATASET_MNIST_1, DATASET_MNIST_2]

DATASET_CONFIG = {
    DATASET_MNIST_1: {
        "name": DATASET_MNIST_1,
        "class-size": 10,
        "train-sizes": [600, 6000, 50000],
        "valid-size": 1000,
        "test-size": 1000,
        "num-splits": 5,
        "num-digits": 1,
        "max-sum": 18,
        "overlaps": [0.0],
    },
    DATASET_MNIST_2: {
        "name": DATASET_MNIST_2,
        "class-size": 10,
        "train-sizes": [600, 6000, 50000],
        "valid-size": 1000,
        "test-size": 1000,
        "num-splits": 5,
        "num-digits": 2,
        "max-sum": 198,
        "overlaps": [0.0],
    }
}

CONFIG_FILENAME = "config.json"


def normalize_images(images):
    (numImages, width, height) = images.shape

    # Flatten out the images into a 1d array.
    images = images.reshape(numImages, width * height)

    # Normalize the greyscale intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(4)

    return images


def digits_to_number(digits):
    number = 0
    for digit in digits:
        number *= 10
        number += digit
    return number


def digits_to_sum(digits, n_digits):
    return digits_to_number(digits[:n_digits]) + digits_to_number(digits[n_digits:])


def generate_split(config, labels, indexes, shuffle=True):
    original_length = len(indexes)
    for _ in range(int(len(indexes) * config['overlap'])):
        indexes = numpy.append(indexes, indexes[numpy.random.randint(0, original_length)])

    if shuffle:
        numpy.random.shuffle(indexes)

    indexes = indexes[:len(indexes) - (len(indexes) % (2 * config['num-digits']))]
    indexes = numpy.unique(indexes.reshape(-1, 2 * config['num-digits']), axis=0)

    sum_labels = numpy.array([digits_to_sum(digits, config['num-digits']) for digits in labels[indexes]])

    return indexes, sum_labels


def create_entity_data_map(features, labels, entities):
    features = normalize_images(features)[entities]
    labels = labels[entities].reshape(-1, 1)
    entities = entities.reshape(-1, 1)

    entity_data_map = numpy.concatenate((entities, features, labels), axis=1).tolist()
    return [[int(row[0])] + row[1:] for row in entity_data_map]


def create_image_digit_sum_data(sum_entities):
    image_digit_sum_targets = []
    for example_indices in sum_entities:
        image_digit_sum_targets += [[example_indices[0], example_indices[2], k] for k in range(19)]
        image_digit_sum_targets += [[example_indices[1], example_indices[3], k] for k in range(19)]
    image_digit_sum_targets = numpy.unique(image_digit_sum_targets, axis=0).tolist()

    return image_digit_sum_targets


def create_image_sum_data(config, sum_entities, sum_labels):
    image_sum_target = []
    image_sum_truth = []
    for index_i in range(len(sum_entities)):
        for index_j in range(config['max-sum'] + 1):
            image_sum_target.append(list(sum_entities[index_i]) + [index_j])
            image_sum_truth.append(list(sum_entities[index_i]) + [index_j] + [1 if index_j == sum_labels[index_i] else 0])

    sum_place_target = []
    sum_place_truth = []
    for entity_index in range(len(sum_entities)):
        if config['num-digits'] == 1:
            places = [1, 10]
        elif config['num-digits'] == 2:
            places = [1, 10, 100]
        else:
            raise Exception("Invalid number of digits.")

        for place in places:
            for z in range(config['class-size']):
                if place == 1:
                    sum_place_target.append(list(sum_entities[entity_index]) + [place] + [z])
                    if config['num-digits'] == 1:
                        sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int(("%02d" % sum_labels[entity_index])[-1]) else 0])
                    elif config['num-digits'] == 2:
                        sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int(("%03d" % sum_labels[entity_index])[-1]) else 0])
                    else:
                        raise Exception("Invalid number of digits.")
                if place == 10:
                    sum_place_target.append(list(sum_entities[entity_index]) + [place] + [z])
                    if config['num-digits'] == 1:
                        sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int(("%02d" % sum_labels[entity_index])[-2]) else 0])
                    elif config['num-digits'] == 2:
                        sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int(("%03d" % sum_labels[entity_index])[-2]) else 0])
                    else:
                        raise Exception("Invalid number of digits.")
                if place == 100:
                    sum_place_target.append(list(sum_entities[entity_index]) + [place] + [z])
                    if config['num-digits'] == 1:
                        raise Exception("Invalid place for number of digits.")
                    elif config['num-digits'] == 2:
                        sum_place_truth.append(list(sum_entities[entity_index]) + [place] + [z] + [1 if z == int(sum_labels[entity_index] >= 100) else 0])
                    else:
                        raise Exception("Invalid number of digits.")

    carry_target = []
    for entity_index in range(len(sum_entities)):
        if config['num-digits'] == 1:
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][1], 0])
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][1], 1])
        elif config['num-digits'] == 2:
            carry_target.append([sum_entities[entity_index][1], sum_entities[entity_index][3], 0])
            carry_target.append([sum_entities[entity_index][1], sum_entities[entity_index][3], 1])
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][2], 0])
            carry_target.append([sum_entities[entity_index][0], sum_entities[entity_index][2], 1])
    carry_target = numpy.unique(carry_target, axis=0).tolist()

    return image_sum_target, image_sum_truth, sum_place_target, sum_place_truth, carry_target


def create_image_data(config, entities):
    image_target = []
    for index_i in range(len(entities)):
        for index_j in range(config['class-size']):
            image_target.append(list(entities[index_i]) + [index_j])

    return image_target


def write_specific_data(config, out_dir, features, labels):
    total_image_entities = numpy.array([], dtype=numpy.int32)

    numpy.random.seed(config['seed'])

    all_indexes = numpy.array(range(len(features)))
    numpy.random.shuffle(all_indexes)

    partition_indexes = {
        'train': all_indexes[0: config['train-size']],
        'valid': all_indexes[config['train-size']: config['train-size'] + config['valid-size']],
        'test': all_indexes[config['train-size'] + config['valid-size']: config['train-size'] + config['valid-size'] + config['test-size']]
    }

    for partition in ['train', 'valid', 'test']:
        image_sum_entities, image_sum_labels = generate_split(config, labels, partition_indexes[partition])
        image_sum_target, image_sum_truth, sum_place_target, sum_place_truth, carry_target = create_image_sum_data(config, image_sum_entities, image_sum_labels)

        image_entities = numpy.unique(image_sum_entities.reshape(-1)).reshape(-1, 1)
        image_target = create_image_data(config, image_entities)

        total_image_entities = numpy.append(total_image_entities, image_entities)

        util.write_psl_data_file(os.path.join(out_dir, f'image-sum-block-{partition}.txt'), image_sum_entities)
        util.write_psl_data_file(os.path.join(out_dir, f'image-sum-place-target-{partition}.txt'), sum_place_target)
        util.write_psl_data_file(os.path.join(out_dir, f'image-sum-place-truth-{partition}.txt'), sum_place_truth)
        util.write_psl_data_file(os.path.join(out_dir, f'image-sum-target-{partition}.txt'), image_sum_target)
        util.write_psl_data_file(os.path.join(out_dir, f'image-sum-truth-{partition}.txt'), image_sum_truth)
        util.write_psl_data_file(os.path.join(out_dir, f'image-target-{partition}.txt'), image_target)
        util.write_psl_data_file(os.path.join(out_dir, f'image-digit-labels-{partition}.txt'),
                                 list(zip(partition_indexes[partition], labels[partition_indexes[partition]])))
        util.write_psl_data_file(os.path.join(out_dir, f'carry-target-{partition}.txt'), carry_target)

    entity_data_map = create_entity_data_map(features, labels, total_image_entities)
    util.write_psl_data_file(os.path.join(out_dir, 'entity-data-map.txt'), entity_data_map)

    util.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def create_sum_data_add1(config):
    possible_digits = []
    for index_i in range(config['class-size']):
        for index_j in range(config['class-size']):
            possible_digits.append([index_i, index_i + index_j])

    digit_sum_ones_place_obs = []
    digit_sum_tens_place_obs = []
    for index_i in range(2):
        for index_j in range(config['class-size']):
            for index_k in range(config['class-size']):
                digit_sum_ones_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) % 10])
                digit_sum_tens_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) // 10])

    placed_representation = []
    for i in range(0, config['max-sum'] + 1):
        representation = "%02d" % i
        placed_representation += [[int(representation[0]), int(representation[1]), i]]

    return possible_digits, digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation


def create_sum_data_add2(config):
    # Possible ones place digits.
    digits_sums = product(range(10), repeat=4)
    possible_ones_digits_dict = {}
    for digits_sum in digits_sums:
        if digits_sum[1] in possible_ones_digits_dict:
            possible_ones_digits_dict[digits_sum[1]].add(
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3])
        else:
            possible_ones_digits_dict[digits_sum[1]] = {
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3]}

    possible_ones_digits = []
    for key in possible_ones_digits_dict:
        for value in possible_ones_digits_dict[key]:
            possible_ones_digits.append([key, value])

    # Possible tens place digits.
    digits_sums = product(range(10), repeat=4)
    possible_tens_digits_dict = {}
    for digits_sum in digits_sums:
        if digits_sum[0] in possible_tens_digits_dict:
            possible_tens_digits_dict[digits_sum[0]].add(
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3])
        else:
            possible_tens_digits_dict[digits_sum[0]] = {
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3]}

    possible_tens_digits = []
    for key in possible_tens_digits_dict:
        for value in possible_tens_digits_dict[key]:
            possible_tens_digits.append([key, value])

    digit_sum_ones_place_obs = []
    digit_sum_tens_place_obs = []
    for index_i in range(2):
        for index_j in range(config['class-size']):
            for index_k in range(config['class-size']):
                digit_sum_ones_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) % 10])
                digit_sum_tens_place_obs.append([index_i, index_j, index_k, (index_i + index_j + index_k) // 10])

    # Placed Representation.
    placed_representation = []
    for i in range(0, config['max-sum'] + 1):
        representation = "%03d" % i
        placed_representation += [[int(representation[0]), int(representation[1]), int(representation[2]), i]]

    return possible_ones_digits, possible_tens_digits, digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation


def write_shared_data(config, out_dir):
    if config['num-digits'] == 1:
        possible_digits,digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation = create_sum_data_add1(config)
        util.write_psl_data_file(os.path.join(out_dir, 'possible-digit-obs.txt'), possible_digits)
        util.write_psl_data_file(os.path.join(out_dir, 'digit-sum-ones-place-obs.txt'), digit_sum_ones_place_obs)
        util.write_psl_data_file(os.path.join(out_dir, 'digit-sum-tens-place-obs.txt'), digit_sum_tens_place_obs)
        util.write_psl_data_file(os.path.join(out_dir, 'placed-representation.txt'), placed_representation)

    elif config['num-digits'] == 2:
        possible_ones_digits, possible_tens_digits, digit_sum_ones_place_obs, digit_sum_tens_place_obs, placed_representation = create_sum_data_add2(config)
        util.write_psl_data_file(os.path.join(out_dir, 'possible-ones-digit-obs.txt'), possible_ones_digits)
        util.write_psl_data_file(os.path.join(out_dir, 'possible-tens-digit-obs.txt'), possible_tens_digits)
        util.write_psl_data_file(os.path.join(out_dir, 'digit-sum-ones-place-obs.txt'), digit_sum_ones_place_obs)
        util.write_psl_data_file(os.path.join(out_dir, 'digit-sum-tens-place-obs.txt'), digit_sum_tens_place_obs)
        util.write_psl_data_file(os.path.join(out_dir, 'placed-representation.txt'), placed_representation)

    else:
        raise Exception(f"Unsupported num-digits: {config['num-digits']}")

    util.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def fetch_data():
    os.makedirs(os.path.join(THIS_DIR, "..", "data", "mnist_raw_data"), exist_ok=True)
    mnist_dataset = torchvision.datasets.MNIST(os.path.join(THIS_DIR, "..", "data", "mnist_raw_data"), download=True)
    return mnist_dataset.data.numpy(), mnist_dataset.targets.numpy()


def main():
    for dataset_id in DATASETS:
        config = DATASET_CONFIG[dataset_id]

        shared_out_dir = os.path.join(THIS_DIR, "..", "data", "experiment::" + dataset_id)
        os.makedirs(shared_out_dir, exist_ok=True)
        if os.path.isfile(os.path.join(shared_out_dir, CONFIG_FILENAME)):
            print("Shared data already exists for %s. Skipping generation." % dataset_id)
        else:
            print("Generating shared data for %s." % dataset_id)
            write_shared_data(config, shared_out_dir)

        for split in range(config['num-splits']):
            for train_size in config['train-sizes']:
                config['train-size'] = train_size
                config['seed'] = 10 * (10 * train_size + split) + config['num-digits']
                print("Using seed %d." % config['seed'])
                for overlap in config['overlaps']:
                    config['overlap'] = overlap
                    out_dir = os.path.join(shared_out_dir, "split::%01d" % split, "train-size::%05d" % train_size, "overlap::%.2f" % overlap)
                    os.makedirs(out_dir, exist_ok=True)

                    if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                        print("Data already exists for %s. Skipping generation." % out_dir)
                        continue

                    print("Generating data for %s." % out_dir)
                    features, labels = fetch_data()
                    write_specific_data(config, out_dir, features, labels)


if __name__ == '__main__':
    main()
