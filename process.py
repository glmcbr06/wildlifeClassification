import os
import logging.config
import h5py
import numpy as np
# Our user made files
from createLabels import *
logger = logging.getLogger(__name__)


def main(args):
    file = args.filepath
    data = {}
    with h5py.File(file, "r") as hf:
        classes = list(hf)
        # idx = np.arange(0, len(classes))
        for cls in classes:
            data[cls] = list(hf[cls]['data'])
    idx = np.arange(0, len(classes))  # to id the classes numerically
    logger.info('The classes included in the data are {}'.format(classes))

    # Gets a matrix of input data and a corresponding vector of target labels
    inputs, targets = createLabels(data)





if __name__ == "__main__":
    """Take the oregon wildlife h5 file and convert into machine readable
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', help='cluster data')
    parser.add_argument('-v', action='store_true', help='Show DEBUG log')

    parsed_args = parser.parse_args()
    if parsed_args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(parsed_args)
