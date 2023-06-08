'''
Features:
- 0. long hair?
- 1. short leg?
- 2. does "au au"?

Classification:
- 0 = dog
- 1 = pig
'''

from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

train_labels = [1, 1, 1, 0, 0, 0]
train_data = [
  [0, 1, 0],
  [0, 1, 1],
  [1, 1, 0], 
  [0, 1, 1],
  [1, 0, 1], 
  [1, 1, 1],
]

model = LinearSVC()
model.fit(train_data, train_labels)

test_labels = [0, 1, 1]
test_data = [
  [1, 1, 1],
  [1, 1, 0],
  [0, 1, 1],
]

predictions = model.predict(test_data)
correct_predictions = (predictions == test_labels).sum()
total = len(test_data)
accuracy = accuracy_score(test_labels, predictions) * 100

print("Accuracy: %.2f%%" % accuracy)