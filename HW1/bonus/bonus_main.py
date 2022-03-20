from bonus_utils import evaluate
from bonus_dataset import loadImages

train_dataset = loadImages("data/train")
test_dataset = loadImages("data/test")

print("Evaluate your classifier with train dataset")
evaluate(train_dataset)
print("Evaluate your classifier with test dataset")
evaluate(test_dataset)