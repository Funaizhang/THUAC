"""
"""
import collections
import random
import time
import data_loader
import evaluation


def assign(data_train):
    """
    """
    assignment = collections.defaultdict(list)
    for authors_id, papers_list in data_train.items():
        author_assignment = [[]]
        for paper in papers_list:
            last_index = len(author_assignment) - 1
            bucket = random.randint(0, last_index)
            author_assignment[bucket].append(paper['id'])
            if bucket == last_index:
                author_assignment.append([])
        author_assignment.pop()
        assignment[authors_id] = author_assignment

    return assignment


def main(samples=3):
    """
    """
    pubs_train = data_loader.load_json_file('pubs_train.json')
    assignment_train = data_loader.load_json_file('assignment_train.json')

    for i in range(samples):
        start = time.time()
        assignment = assign(pubs_train)
        print("Model trained in {:.3f} secs".format(time.time() - start))

        data_loader.save_json(
            "assignment_validate_random_{}.json".format(i),
            assignment)

        print("Pairwise F1: {:.5f}".format(
            evaluation.pairwise_f1(assignment, assignment_train)))


if __name__ == "__main__":
    main()
