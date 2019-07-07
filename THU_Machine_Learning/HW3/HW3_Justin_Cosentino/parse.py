"""
Loads and tags documents.
"""
import time
import pickle
import os
import data_loader
import data_parser
import constants


def _build_and_save_tagged_docs(
        file_name,
        path=constants.DATA_DIR,
        keys=constants.KEYS,
        use_prefix=True):
    data_train = data_loader.load_json_file('{}.json'.format(file_name))
    tagged_docs = data_parser.build_tagged_docs(
        data_train,
        keys=keys,
        use_prefix=use_prefix)
    with open(_build_path(file_name, path=path), 'wb') as fp:
        pickle.dump(tagged_docs, fp)
    return tagged_docs


def load_tagged_docs(
        file_name,
        path=constants.DATA_DIR,
        force_build=False,
        keys=constants.KEYS,
        use_prefix=True):
    start = time.time()
    tagged_docs = None
    if not force_build and check_tagged_docs(file_name, path=path):
        with open(_build_path(file_name, path=path), 'rb') as fp:
            tagged_docs = pickle.load(fp)
    else:
        tagged_docs = _build_and_save_tagged_docs(
            file_name,
            path=path,
            keys=keys,
            use_prefix=use_prefix)
    print("Total docs: {} ({:.3f} secs)".format(
        len(tagged_docs),
        time.time() - start))
    return tagged_docs


def check_tagged_docs(file_name, path=constants.DATA_DIR):
    return os.path.isfile(_build_path(file_name, path=path))


def _build_path(file_name, path=constants.DATA_DIR):
    return './{}/{}_tagged_docs'.format(path, file_name)


def main():
    load_tagged_docs('pubs_train', keys=constants.KEYS_WITHOUT_ABSTRACT)


if __name__ == '__main__':
    main()
