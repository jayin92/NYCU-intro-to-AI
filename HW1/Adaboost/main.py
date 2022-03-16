import dataset
import adaboost
import utils
import detection
import matplotlib.pyplot as plt
import argparse

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-t",
                    type=int, default=10)
args = parser.parse_args()

# Part 1: Implement loadImages function in dataset.py and test the following code.
print('Loading images')
trainData = dataset.loadImages('data/train')
print(f'The number of training samples loaded: {len(trainData)}')
testData = dataset.loadImages('data/test')
print(f'The number of test samples loaded: {len(testData)}')

# print('Show the first and last images of training dataset')
# fig, ax = plt.subplots(1, 2)
# ax[0].axis('off')
# ax[0].set_title('Car')
# ax[0].imshow(trainData[1][0], cmap='gray')
# ax[1].axis('off')
# ax[1].set_title('Non car')
# ax[1].imshow(trainData[-1][0], cmap='gray')
# plt.show()

# Part 2: Implement selectBest function in adaboost.py and test the following code.
# Part 3: Modify difference values at parameter T of the Adaboost algorithm.
# And find better results. Please test value 1~10 at least.



clfs = []
# t = args.t
print('Start training your classifier')
for t in range(1, 11): # Train with different T (1 ~ 10)
    print(f"Training with T={t}") 
    clf = adaboost.Adaboost(T=t) # Init. clf using the given T
    clf.train(trainData) # Train of train data
    clf.save(f'clf_300_{t}') # Save the model file as clf_300_<t>
    clf = adaboost.Adaboost.load(f'clf_300_{t}') # Load the model after saving

    # Evaluate the model using already written utils.evaluate function
    print('\nEvaluate your classifier with training dataset')
    utils.evaluate(clf, trainData) # Use train data to evaluate model

    print('\nEvaluate your classifier with test dataset')
    utils.evaluate(clf, testData) # Use test data to evaluate model

    # Part 4: Implement detect function in detection.py and test the following code.
    print('\nUse your classifier with video.gif to get the predictions (one .txt and one .png)')
    detection.detect('data/detect/detectData.txt', clf, t) # Save the detectData.txt

"""
clf = adaboost.Adaboost(T=10)
clf.train(trainData)

clf.save('clf_300_1_10_2_norm')
clf = adaboost.Adaboost.load('clf_300_1_10_2_norm')

print('\nEvaluate your classifier with training dataset')
utils.evaluate(clf, trainData)

print('\nEvaluate your classifier with test dataset')
utils.evaluate(clf, testData)

# Part 4: Implement detect function in detection.py and test the following code.
print('\nUse your classifier with video.gif to get the predictions (one .txt and one .png)')
detection.detect('data/detect/detectData.txt', clf)
"""

