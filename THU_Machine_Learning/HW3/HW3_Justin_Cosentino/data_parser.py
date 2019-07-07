"""
Used to parse JSON, strings, and build features.
"""
import string
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
import constants

STOPWORDS_SET = set(stopwords.words('english'))


def norm_text(text, replace_punct_with=constants.EMPTY):
    """
    Clean a single word, returning a string.
    """
    if not text:
        return ""

    text = str(text).strip().lower()
    for punct in string.punctuation:
        text = text.replace(punct, replace_punct_with)
    return text


def norm_name(name):
    """
    Clean a name, replacing puncts and spaces with "_", returning a string.
    """
    if not name:
        return ""

    name = str(name).strip().lower()
    if name[-1] == "-":
        name = name[:-1]
    if name[0] == "-":
        name = name[1:]
    if name[-1] == ".":
        name = name[:-1]
    name = name.replace("-", constants.UNDERSCORE)
    name = name.replace("..", constants.UNDERSCORE)
    name = name.replace(".", constants.UNDERSCORE)
    name = name.replace(constants.SPACE, constants.UNDERSCORE)
    name = name.replace(
        constants.UNDERSCORE*2,
        constants.UNDERSCORE)
    return name


def norm_sent_to_list(text, use_stopwords=False):
    """
    Clean each word in a sentence, removing stopwords and returning a list.
    """
    if not text:
        return []

    if not use_stopwords:
        [w for w in norm_text(text).split() if w not in STOPWORDS_SET]

    return norm_text(text).split()


def norm_sent_to_sent(text, use_stopwords=False):
    """
    Clean each word in a sentence, removing stopwords and returning a string.
    """
    if not text:
        return ""

    if not use_stopwords:
        return " ".join(
            [w for w in norm_text(text).split() if w not in STOPWORDS_SET])

    return norm_text(text)


def convert_text_to_feature(feature_name, text_list):
    """
    Converts a text list to a list of features, with each text list item
    prefixed by the feature name.
    """
    assert isinstance(text_list, list)
    feature_name = feature_name.upper()
    return ["__{}__{}".format(feature_name, text) for text in text_list]


def parse_authors(authors, use_prefix):
    features = []
    for i, author in enumerate(authors):
        names = []
        orgs = []
        co_name = [norm_name(author['name'])]
        co_org = [norm_sent_to_sent(author['org'])]
        names += convert_text_to_feature("name", co_name) if use_prefix else co_name
        orgs += convert_text_to_feature("org", co_org) if use_prefix else co_org
        features += names + orgs
    return features


def parse_keywords(kws, use_prefix=True):
    """
    Returns a feature representation list for keywords.
    """
    kws = [norm_text(k) for k in kws if k != '' and k not in STOPWORDS_SET]
    if use_prefix:
        return convert_text_to_feature('keywords', kws)
    return kws


def parse_title(title, use_prefix=True):
    """
    Returns a feature representation list for title.
    """
    if use_prefix:
        return convert_text_to_feature('title', norm_sent_to_list(title))
    return norm_sent_to_list(title)


def parse_venue(venue, use_prefix=True):
    """
    Returns a feature representation list for venue.
    """
    if use_prefix:
        return convert_text_to_feature('venue', norm_sent_to_list(venue))
    return norm_sent_to_list(venue)


def parse_year(year, use_prefix=True):
    """
    Returns a feature representation list for year.
    """
    if use_prefix:
        return convert_text_to_feature('year', norm_sent_to_list(year))
    return norm_sent_to_list(year)


def convert_paper_to_doc(paper, keys=constants.KEYS, use_prefix=True):
    doc = []
    for key in keys:
        if key == 'id':
            continue

        if key not in paper:
            continue

        if key == 'authors':
            doc += parse_authors(paper[key], use_prefix=use_prefix)

        elif key == "keywords":
            doc += parse_keywords(paper[key], use_prefix=use_prefix)

        elif key == 'title':
            doc += parse_title(paper[key], use_prefix=use_prefix)

        elif key == 'venue':
            doc += parse_venue(paper[key], use_prefix=use_prefix)

        elif key == 'year':
            doc += parse_year(paper[key], use_prefix=use_prefix)

        else:
            val = norm_sent_to_list(paper[key])
            doc += val

    return doc


def build_doc_tuples(data_train, keys=constants.KEYS, use_prefix=True):
    docs = []
    for authors_id, papers_list in data_train.items():
        for paper in papers_list:
            docs.append((paper['id'], convert_paper_to_doc(
                paper,
                keys=keys,
                use_prefix=use_prefix)))
    return docs


def build_tagged_docs(papers, keys=constants.KEYS, use_prefix=True):
    docs = build_doc_tuples(papers, keys=keys, use_prefix=use_prefix)
    return [TaggedDocument(doc[1], [doc[0]]) for i, doc in enumerate(docs)]
