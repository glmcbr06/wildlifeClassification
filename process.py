from keras.models import Sequential
# import tensorflow
import os
import logging.config
import h5py
from createLabels import *
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def main(args):

    np.random.seed(1)
    directory = args.directory

    dataDir = os.path.join(directory, 'data')
    plotsDir = os.path.join(directory, 'plots')
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
        logger.info('Making the data directory...')
    else:
        logger.info('Data directory already exists...continuing')
    if not os.path.exists(plotsDir):
        os.mkdir(plotsDir)
        logger.info('Making the plots directory...')
    else:
        logger.info('Plots directory already exists...continuing')

    file = args.filePath
    if args.trainingPercent:
        pct = args.trainingPercent
    else:
        pct = .6

    if not args.epochs:
        epochs = 30
        logger.info('running with {} epochs'.format(epochs))
    else:
        epochs = args.epochs
        logger.info('running with {} epochs'.format(epochs))

    data = {}
    with h5py.File(file, "r") as hf:
        classes = list(hf)
        for cls in classes:
            data[cls] = list(hf[cls]['data'])
    logger.info('The classes included in the data are {}'.format(classes))
    # Gets a matrix of input data and a corresponding vector of target labels
    inputs, targets = createLabels(data)
    n = inputs.shape[0]
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
    trainHistory = model.fit(xTrain, yTrain, validation_data=(xTest[300:, :], yTest[300:, :]), epochs=50)

    # todo: use a validation set if time....not using predict_classes right.
    pred = model.predict_classes(xTest[:300, :])
    actual = [np.argmax(val) for val in yTest[:300, :]]
    success = 0

    for i in range(len(actual)):
        p = pred[i]
        a = actual[i]
        if p == a:
            success += 1
    acc = success / len(actual)
    print('Validation accuracy: ', acc)


    trainMetricsDf = pd.DataFrame(trainHistory.history)
    fname = os.path.join(dataDir, 'modelMetrics.csv')
    trainMetricsDf.to_csv(fname)

    f, ax = plt.subplots()
    epochs = list(trainMetricsDf.index)
    ax.plot(epochs, trainMetricsDf.accuracy, label='Training Accuracy')
    ax.plot(epochs, trainMetricsDf.val_accuracy, label='Test Accuracy')
    plt.legend()
    plt.title('Training vs. Test Accuracy. Epochs: {}'.format(len(epochs)))
    fname = os.path.join(plotsDir, 'accuracyPlot.png')
    f.savefig(fname)

if __name__ == "__main__":
    """Take the oregon wildlife h5 file and convert into machine readable
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--filePath', help='cluster data')
    parser.add_argument('--directory', required=True, help='the directory to save your plots and data files')
    parser.add_argument('--epochs', help='number of epochs to run')
    parser.add_argument('--trainingPercent', help='choose the training percent, as a decimal')
    parser.add_argument('-v', action='store_true', help='Show DEBUG log')


    parsed_args = parser.parse_args()
    if parsed_args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    main(parsed_args)
