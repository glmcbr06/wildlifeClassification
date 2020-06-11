from keras.models import Sequential
import os
import logging.config
import h5py
from createLabels import *

from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical

logger = logging.getLogger(__name__)


def main(args):

    np.random.seed(1)
    file = args.filePath
    if args.trainingPercent:
        pct = args.trainingPercent
    else:
        pct = .6

    data = {}
    with h5py.File(file, "r") as hf:
        classes = list(hf)
        # idx = np.arange(0, len(classes))
        for cls in classes:
            data[cls] = list(hf[cls]['data'])
    logger.info('The classes included in the data are {}'.format(classes))
    # Gets a matrix of input data and a corresponding vector of target labels
    inputs, targets = createLabels(data)
    print(inputs.shape)
    exit()

    targets = np.array(targets).reshape(inputs.shape[0], 1)
    npData = np.hstack((inputs, targets))
    np.random.shuffle(npData)
    n = npData.shape[0]
    rows = np.random.randint(n, size=int(n*pct))
    train = npData[rows, :]

    x, y = train[:, :train.shape[1] - 1], train[:, train.shape[1] - 1]
    print(x.shape)
    exit()




if __name__ == "__main__":
    """Take the oregon wildlife h5 file and convert into machine readable
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filePath', help='cluster data')
    parser.add_argument('--trainingPercent', help='choose the training percent')
    parser.add_argument('-v', action='store_true', help='Show DEBUG log')


    parsed_args = parser.parse_args()
    if parsed_args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(parsed_args)
