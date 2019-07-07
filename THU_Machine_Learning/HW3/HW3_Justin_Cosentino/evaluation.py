import collections
import itertools
import data_loader
from itertools import combinations
import multiprocessing as mp
import datetime
from zipfile import *
from os import listdir
from os.path import isfile, join


def _generate_pairwise_sets(assignment):
    """
    """
    author_pairs = collections.defaultdict(set)
    for author_id, paper_groups in assignment.items():
        pairs = set()
        for group in paper_groups:
            for pair in itertools.combinations(group, 2):
                pairs.add(pair)
        author_pairs[author_id] = pairs
    return author_pairs


def _pairwise_precision(inter, predictions_pairs):
    """
    """
    if not predictions_pairs:
        return 0
    return len(inter) / len(predictions_pairs)


def _pairwise_recall(inter, groundtruth_pairs):
    """
    """
    if not groundtruth_pairs:
        return 0
    return len(inter) / len(groundtruth_pairs)


def pairwise_f1(predictions, groundtruth):
    """
    """
    # predictions_pairs = _generate_pairwise_sets(predictions)
    # groundtruth_pairs = _generate_pairwise_sets(groundtruth)

    # f1s = []
    # for author_id in groundtruth_pairs.keys():
    #     preds_author_pairs = predictions_pairs[author_id]
    #     truth_author_pairs = groundtruth_pairs[author_id]
    #     inter = preds_author_pairs.intersection(truth_author_pairs)
    #     precision = _pairwise_precision(inter, preds_author_pairs)
    #     recall = _pairwise_recall(inter, truth_author_pairs)
    #     if precision + recall == 0:
    #         f1s.append(0)
    #     else:
    #         f1 = (2 * precision * recall) / (precision + recall)
    #         f1s.append(f1)
    #     print("Precision: {:.3f} Recall: {:.3f}".format(precision, recall))
    # return sum(f1s) / len(f1s)
    sub = predictions
    truth = groundtruth
    output = mp.Queue()

    count = 0
    paper_set = set([])
    truth_pair = {}
    for item in truth:
        truth_pair[item] = []
        for i in range(len(truth[item])):
            set_list = [set(p) for p in combinations(truth[item][i], 2)]
            truth_pair[item] += set_list
            count += len(set_list)
            for p in truth[item][i]:
                    paper_set.add(p)

    sub_pair = {}
    c = 0
    for item in sub:
        sub_pair[item] = []
        for i in range(len(sub[item])):
            set_list = [set(p) for p in combinations(sub[item][i], 2) if p[0] in paper_set and p[1] in paper_set]
            sub_pair[item] += set_list
            c += len(set_list)

    part = [{},{},{}]
    for n, item in enumerate(sub_pair):
        if item == 't_suzuki':
            part[0][item] = sub_pair[item]
        elif item == 'q_liu':
            part[1][item] = sub_pair[item]
        else:
            part[2][item] = sub_pair[item]

    def count_part(p, output):
        correct = 0
        for name in p:
            for item in p[name]:
                if item in truth_pair[name]:
                    correct += 1
        output.put(correct)
        return correct

    processes = [mp.Process(target=count_part, args=(part[x], output)) for x in range(3)]

    begin = datetime.datetime.now()
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = [output.get() for p in processes]
    end = datetime.datetime.now()

    correct = 0
    for i in results:
        correct += i

    p = correct / c if c != 0 else 0
    r = correct / count

    if p + r == 0:
        score = 0
    else:
        score = 2 * p * r /(p + r)
    print("Precision: {:.3f} Recall: {:.3f}".format(p, r))
    return score


def main():
    """
    """
    predictions = data_loader.load_json_file(
        'assignment_validate_random_1.json',
        path='./data/output')
    groundtruth = data_loader.load_json_file('assignment_validate.json')
    print(pairwise_f1(predictions, groundtruth))


if __name__ == '__main__':
    main()
