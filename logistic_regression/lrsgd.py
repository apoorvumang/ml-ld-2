import numpy as np
import math

ALPHA = 0.01
LAMBDA = 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)

def cost_function(features, labels, weights):
    '''
    Using Mean Absolute Error

    Features:(100,3)
    Labels: (100,1)
    Weights:(3,1)
    Returns 1D matrix of predictions
    Cost = ( log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)

    predictions = predict(features, weights)

    #Take the error when label=1
    class1_cost = -labels*np.log(predictions)

    #Take the error when label=0
    class2_cost = (np.ones(len(labels))-labels)*np.log(np.ones(len(predictions))-predictions)

    #Take the sum of both costs
    cost = class1_cost - class2_cost

    #Take the average cost
    cost = cost.sum()/observations

    return cost


def update_weights(features, labels, weights, lr):
    '''
    Vectorized Gradient Descent

    Features:(200, 3)
    Labels: (200, 1)
    Weights:(3, 1)
    '''
    N = len(features)

    #1 - Get Predictions
    predictions = predict(features, weights)

    #2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(features.T,  predictions - labels)

    #3 Take the average cost derivative for each feature
    gradient /= N

    #4 - Multiply the gradient by our learning rate
    gradient *= lr

    #5 - Subtract from our weights to minimize cost
    weights -= gradient
    weights *= (1-LAMBDA)
    return weights 

def decision_boundary(prob):
    return 1 if prob >= .5 else 0

def classify(predictions):
    db = np.vectorize(decision_boundary)
    return db(predictions).flatten()


def train(features, labels, weights, lr, iters):
    cost_history = []

    train_features = np.array(features[:80])
    train_labels = labels[:80]
    test_features = np.array(features[80:])
    test_labels = labels[80:]
    for i in range(iters):
        for j in range(len(train_features)):
            instance = train_features[j]
            label = train_labels[j]
            weights = update_weights(instance, label, weights, lr)
        #Calculate error for auditing purposes
        cost = cost_function(train_features, train_labels, weights)
        cost_history.append(cost)
        # Log Progress
        if i % 10 == 0:
            predicted_probs = predict(test_features, weights)
            predicted_labels = classify(predicted_probs)
            acc_test = accuracy(predicted_labels, test_labels)
            acc_train = accuracy(classify(predict(train_features, weights)), train_labels)

            print "iter: "+str(i) + " cost: "+str(cost) + " acc train: "+str(acc_train) + " accuracy test: " + str(acc_test)

    return weights, cost_history

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

def plot_decision_boundary(trues, falses):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    no_of_preds = len(trues) + len(falses)

    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

    plt.legend(loc='upper right');
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()

file = open("data.txt", "r")

data = []
labels = []

for line in file.readlines():
    line = line.strip()
    splitLine = line.split(",")
    labels.append(int(splitLine[2]))
    instance = []
    instance.append(float(splitLine[0]))
    instance.append(float(splitLine[1]))
    instance.append(float(1))
    data.append(instance)

weights = np.array([0.0, 0.0, 0.0])
data = np.array(data)
labels = np.array(labels)
weights, cost_history = train(data, labels, weights, ALPHA, 500)

predicted_probs = predict(data, weights)
predicted_labels = classify(predicted_probs)

print accuracy(predicted_labels, labels)
print(weights)




















