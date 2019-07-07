"""
Cluster based on org with learned word embeddings to handle off by ones and
acronyms.
"""
import collections
import random
import time
import data_loader
import data_parser
import evaluation
import parse
from gensim.models.doc2vec import TaggedDocument
import d2vec_model
import constants


def get_author_org(paper, authors_id):
    for author in paper['authors']:
        current_id = data_parser.norm_name(author['name'])
        if authors_id == current_id:
            return data_parser.norm_sent_to_sent(author['org'])
    print(authors_id, paper)
    raise Exception("Did not find author for paper!")


def assign(data_train, tagged_docs_file_name):
    """
    """
    last = ''
    orgs_all = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for authors_id, papers_list in data_train.items():
        orgs = collections.defaultdict(list)
        for paper in papers_list:
            department = get_author_org(paper, authors_id)
            if department and department[-1] == 's':
                department = department[:-1]
            if department == '':
                department = last
            elif department in last and last != '':
                department = last
            elif last in department and last != '':
                department = last
            last = department
            orgs[department].append(paper['id'])
        orgs_all[authors_id] = orgs

    loaded_tagged_docs = parse.load_tagged_docs(
        tagged_docs_file_name,
        keys=['authors'],
        use_prefix=False,
        force_build=False)
    new_loaded_tagged_docs = []
    org_to_doc = collections.defaultdict(list)
    for doc in loaded_tagged_docs:
        doc_set = set()
        for word in doc.words:
            if '_' not in word:
                doc_set.add(word)
        new_loaded_tagged_docs += [TaggedDocument(org, doc.tags) for org in doc_set]
        for org in doc_set:
            org_to_doc[org].extend(doc.tags)
    loaded_tagged_docs = new_loaded_tagged_docs

    # Build tagged docs and train model for each author
    print("Tagging docs and training author models.")
    author_tagged_docs = collections.defaultdict(list)
    all_tagged_docs = []
    for authors_id in data_train:
        orgs = orgs_all[authors_id].keys()
        tagged_docs = [TaggedDocument(org, [i]) for i, org in enumerate(orgs)]
        author_tagged_docs[authors_id] = tagged_docs
        all_tagged_docs += tagged_docs
    d2vec = d2vec_model.load_maybe_train(
        all_tagged_docs + loaded_tagged_docs,
        path="{}/{}_{}".format(
            constants.MODEL_DIR,
            tagged_docs_file_name,
            constants.D2V_MODEL_FILE_NAME),
        force_train=True
        )

    # Use models to determine which should be merged
    print("Determining merge sets.")
    to_merge = collections.defaultdict(set)
    for authors_id in data_train:
        tagged_docs = author_tagged_docs[authors_id]
        for i, doc_a in enumerate(tagged_docs[:-1]):
            for j, doc_b in enumerate(tagged_docs[i+1:]):
                if not doc_a.words or not doc_b.words:
                    continue
                if d2vec.n_similarity(doc_a.words, doc_b.words) >= 1:
                    to_merge[authors_id].add((doc_a.words, doc_b.words))

    for authors_id in data_train:
        deleted = set()
        for item in to_merge[authors_id]:
            if item[0] in deleted:
                print("already deleted!", item[0])
            elif item[1] in deleted:
                print("already deleted!", item[1])
            else:
                orgs_all[authors_id][item[0]] += orgs_all[authors_id][item[1]]
                del orgs_all[authors_id][item[1]]
                deleted.add(item[1])

    result = collections.defaultdict(list)
    for authors_id in data_train:
        result[authors_id] = [ids for _, ids in orgs_all[authors_id].items()]
    return result


def main():
    """
    """
    data_src = 'test'
    train_data = data_loader.load_json_file('pubs_{}.json'.format(data_src))
    # groundtruth = data_loader.load_json_file('assignment_{}.json'.format(data_src))
    start = time.time()
    assignment = assign(train_data, 'pubs_{}'.format(data_src))
    print("Model trained in {:.3f} secs".format(time.time() - start))
    data_loader.save_json("assignment_{}_org.json".format(data_src), assignment)
    # print("Pairwise F1: {:.5f}".format(
    #     evaluation.pairwise_f1(assignment, groundtruth)))


if __name__ == "__main__":
    main()
