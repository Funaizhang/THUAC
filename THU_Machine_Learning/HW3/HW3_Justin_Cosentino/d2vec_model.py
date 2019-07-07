"""
Trains and saves doc2vec models.
"""
import os.path
import gensim
from gensim.models.doc2vec import Doc2Vec
import constants


def load_maybe_train(
        tagged_docs,
        save=True,
        workers=6,
        force_train=False,
        path="{}/{}".format(
            constants.MODEL_DIR,
            constants.D2V_MODEL_FILE_NAME)):

    if force_train:
        print("Force training a new model.")
        return build(tagged_docs, save=save, workers=workers, path=path)

    if os.path.isfile(path):
        print("Model already exists at {}. Loading.".format(path))
        return Doc2Vec.load(path)

    print("Model does not exist at {}. Training.".format(path))
    return build(tagged_docs, save=save, workers=workers, path=path)


def build(
        tagged_docs,
        save=True,
        workers=6,
        path="{}/{}".format(
            constants.MODEL_DIR,
            constants.D2V_MODEL_FILE_NAME)):
    """
    """
    d2v_model = Doc2Vec(
        tagged_docs,
        epochs=20,
        min_count=1,
        vector_size=100,
        workers=workers)
    d2v_model.delete_temporary_training_data(
        keep_doctags_vectors=True,
        keep_inference=True)

    if save:
        print("Saving model at {}".format(path))
        d2v_model.save(path)

    return d2v_model
