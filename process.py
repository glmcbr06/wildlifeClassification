from keras.models import Sequential
# import tensorflow
import os
import logging.config
import h5py
from createLabels import *
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical

logger = logging.getLogger(__name__)


def setdiff_sorted(array1,array2,assume_unique=False):
    ans = np.setdiff1d(array1,array2,assume_unique).tolist()
    if assume_unique:
        return sorted(ans)
    return ans

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
    n = inputs.shape[0]
    pct = .6
    allRows = np.arange(0, inputs.shape[0])
    np.random.shuffle(allRows)
    inputs = inputs[allRows, :]
    t = []
    for i in allRows:
        t.append(targets[i])
    xTrain, xTest = inputs[:int(n*pct), :], inputs[int(n*pct):, :]
    yTrain, yTest = t[:int(n*pct)], t[int(n*pct):]

    print(xTrain.shape, xTest.shape)
    print(len(yTrain), len(yTest))

    yTrain = to_categorical(yTrain)
    yTest = to_categorical(yTest)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=3)
    model.predict(xTest[:4])


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
