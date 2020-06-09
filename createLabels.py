import numpy as np


""" The labels correspond to the following animals:
0 --> bald eagle
1 --> black bear
2 --> cougar
3 --> elk
4 --> gray wolf
"""

# A function that take a dictionary of data points and returns an input matrix
# as well as a vector of corresponding targets
def createLabels(data):
    inputs = []
    targets = []
    target_class = 0
    for animal in data:
        animal_samples = np.array(data[animal])
        inputs.append(animal_samples)
        temp = [target_class] * (animal_samples.shape[0])
        targets.append(temp)
        target_class += 1

    inputs = np.vstack(inputs)
    return inputs, targets
