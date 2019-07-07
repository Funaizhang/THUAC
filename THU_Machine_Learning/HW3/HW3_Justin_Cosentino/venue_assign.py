"""
Cluster based on venue
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


def assign(data_train):
    """
    """
    last = ''
    clusters_all = collections.defaultdict(list)
    for authors_id, papers_list in data_train.items():
        clusters = []
        for paper in papers_list[::-1]:
            org = get_author_org(paper, authors_id)
            if org and org[-1] == 's':
                org = org[:-1]
            if org == '':
                org = last
            elif org in last and last != '':
                org = last
            elif last in org and last != '':
                org = last
            last = org
            venue = get_paper_venue(paper)
            current = {
                'org': org,
                'venue': venue,
                'id': paper['id']
            }
            added = False
            for cluster in clusters:
                for item in cluster:
                    if item['venue'] == current['venue'] or item['org'] == current['org']:
                        cluster.append(current)
                        added = True
                    if added:
                        break
                if added:
                    break

            if not added:
                clusters.append([current])
                org = get_author_org(paper, authors_id)
                venue = get_paper_venue(paper)
        clusters_all[authors_id] = clusters

    results_all = collections.defaultdict(list)
    for authors_id in data_train:
        clusters_id = []
        for cluster in clusters_all[authors_id]:
            clusters_id.append([item['id'] for item in cluster])
        results_all[authors_id] = clusters_id
    return results_all


def main():
    data_src = 'test'
    train_data = data_loader.load_json_file('pubs_{}.json'.format(data_src))
    # groundtruth = data_loader.load_json_file('assignment_{}.json'.format(data_src))
    start = time.time()
    assignment = assign(train_data)
    print("Model trained in {:.3f} secs".format(time.time() - start))
    data_loader.save_json("assignment_{}_venue.json".format(data_src), assignment)
    # print("Pairwise F1: {:.5f}".format(
    #     evaluation.pairwise_f1(assignment, groundtruth)))


if __name__ == "__main__":
    main()
