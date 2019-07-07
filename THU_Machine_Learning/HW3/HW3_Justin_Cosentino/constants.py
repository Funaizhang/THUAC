"""
Constants used in the project.
"""

# Keys on author JSON blobs
KEYS = [
    'abstract',
    'authors',
    'id',
    'keywords',
    'title',
    'venue',
    'year'
]
KEYS_WITHOUT_ABSTRACT = [
    'authors',
    'keywords',
    'title',
    'venue',
    'year'
]
ABSTRACT = [
    'abstract'
]

# Directories used for data processing
DATA_DIR = './data'
MODEL_DIR = "./models"
D2V_MODEL_FILE_NAME = "word2vec.model"
OUTPUT_DIR = '{}/output'.format(DATA_DIR)

# Constants used in data parsing
EMPTY = ""
SPACE = " "
UNDERSCORE = "_"
