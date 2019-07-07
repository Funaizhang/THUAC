"""
Cluster based on grouping of first seen venue or organization.
"""
import collections
import time
import data_loader
import data_parser
import evaluation


def get_author_org(paper, authors_id):
    for author in paper['authors']:
        current_id = data_parser.norm_name(author['name'])
        if authors_id == current_id:
            return data_parser.norm_sent_to_sent(author['org'])
    print(authors_id, paper)
    raise Exception("Did not find author for paper!")


def get_paper_venue(paper):
    return data_parser.norm_sent_to_sent(paper['venue'])


def assign(data_train, tagged_docs_file_name):
    last = ''
    orgs_all = collections.defaultdict(
        lambda: collections.defaultdict(list))
    venues_all = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for authors_id, papers_list in data_train.items():
        orgs = collections.defaultdict(list)
        venues = collections.defaultdict(list)
        for paper in papers_list:
            org = get_author_org(paper, authors_id)
            venue = get_paper_venue(paper)
            if org and org[-1] == 's':
                org = org[:-1]
            if org == '':
                org = last
            elif org in last and last != '':
                org = last
            elif last in org and last != '':
                org = last
            last = org

            if venue not in orgs:
                orgs[org].append(venue)
            orgs[org].append(paper['id'])
            venues[venue].append(paper['id'])
        orgs_all[authors_id] = orgs
        venues_all[authors_id] = venues

    result = collections.defaultdict(list)
    for authors_id in data_train:
        venues = venues_all[authors_id]
        orgs = orgs_all[authors_id]
        for org, id_list in orgs.items():
            venue, id_list = id_list[0], id_list[1:]
            new_ids_list = list(set(id_list + venues[venue]))
            result[authors_id].append(new_ids_list)
    return result


def main():
    data_src = 'test'
    train_data = data_loader.load_json_file('pubs_{}.json'.format(data_src))
    # groundtruth = data_loader.load_json_file('assignment_{}.json'.format(data_src))
    start = time.time()
    assignment = assign(train_data, 'pubs_{}'.format(data_src))
    print("Model trained in {:.3f} secs".format(time.time() - start))
    data_loader.save_json("assignment_{}_org_or_venue.json".format(data_src), assignment)
    # print("Pairwise F1: {:.5f}".format(
    #     evaluation.pairwise_f1(assignment, groundtruth)))


if __name__ == "__main__":
    main()
