"""
Loads and saves JSON blobs.
"""
import json
import constants


def load_json_file(file_name, path=constants.DATA_DIR):
    file_path = "{}/{}".format(path, file_name)
    print('Loading {}'.format(file_path))
    with open(file_path, 'r') as file:
        return json.load(file, encoding="utf-8")


def save_json(file_name, data, path=constants.OUTPUT_DIR):
    file_path = "{}/{}".format(path, file_name)
    print('Saving {}'.format(file_path))
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
