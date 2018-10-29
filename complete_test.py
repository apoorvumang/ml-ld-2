#!/usr/bin/python

# We need to match exact class predictions
# ie if an instance has correct classes [c1, c2, c3]
# then prediction of c1, c2, c3 should be 1 and rest should be 0
# If this is true, then numCorrect += 1 else numWrong += 1

# Can't load all parameters at once, so we will make list of predictions for each class
# then exact compare between the predicted set and true set
import re
import sys
import json
import numpy as np
import math
import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]

FULL = False
DATA_TYPE = "test"
if len(arguments) == 2:
	if(arguments[0]=="full"):
		FULL = True
	if(arguments[1]=="train"):
		DATA_TYPE = "train"
else:
	print "Need 2 arguments: full/verysmall, test/train"
	exit()

CLASSES_FILE_NAME = "classes.txt"

VOCAB_FILE_NAME = "vocab_verysmall.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse_"+DATA_TYPE+"_verysmall.txt"
if FULL:
	VOCAB_FILE_NAME = "vocab_full.txt"
	DATA_VECTORS_FILE_NAME = "data/vectors_sparse_"+DATA_TYPE+"_full.txt"




vocab = {}
f = open(VOCAB_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    word = splitLine[0]
    wordId = int(splitLine[1])
    vocab[word] = wordId
f.close()

VOCAB_SIZE = len(vocab)


def unpack_vector(sparse_vector, length):
	v = np.zeros((length,), dtype=np.float)
	for key, value in sparse_vector.items():
		key = int(key)
		value = float(value)
		v[key] = value
	return v

def sigma(z):
	value = 0.0
	try:
		value = 1.0/(1.0 + math.exp(-z))
	except OverflowError:
		value = 0.0
	return value

def sparse_mult(normal, sparse):
	ans = 0.0
	for key, value in sparse.items():
		key = int(key)
		value = float(value)
		ans = ans + normal[key]*value
	return ans

def sparse_scalar_mult(sparse, k):
	sparse2 = sparse.copy()
	for key in sparse2:
		sparse2[key] *= k
	return sparse2

def sparse_subtract(normal, sparse):
	for key, value in sparse.items():
		key = int(key)
		value = float(value)
		normal[key] -= value
	return normal


# parameters
W = np.zeros((VOCAB_SIZE,), dtype=np.float)
b = 0
vsigma = np.vectorize(sigma)

# sparse vector data with class
data = []
file = open(DATA_VECTORS_FILE_NAME, "r")
NUM_INSTANCE_TO_PROCESS = 3000000 # if this exceeds no. of lines in file its still ok
count = 0
for line in file.readlines():
	line = json.loads(line)
	data.append(line)
	count = count + 1
	if(count > NUM_INSTANCE_TO_PROCESS):
		break
NUM_INSTANCE_TO_PROCESS = count
file.close()


numCorrect = 0.0
numWrong = 0.0
predicted_classes = {}

simplePredictionA = []
simplePredictionClass = []

for i in range(0,NUM_INSTANCE_TO_PROCESS):
	simplePredictionA.append(0.0)
	simplePredictionClass.append(0)


for class_number in range(0, 50):
	OUTPUT_W_FILE_NAME = "output_multi/output_w"+str(class_number)+".txt"
	OUTPUT_B_FILE_NAME = "output_multi/output_b"+str(class_number)+".txt"
	OUTPUT_PARAMS_FILE_NAME = "output_multi/output_params"+str(class_number)+".txt"
	OUTPUT_HISTORY_FILE_NAME = "output_multi/output_history"+str(class_number)+".txt"
	params = {}
	paramsFile = open(OUTPUT_PARAMS_FILE_NAME, "r")
	for line in paramsFile.readlines():
		if line.strip():
			splitLine = line.strip().split(':')
			key = splitLine[0].strip()
			value = float(splitLine[1].strip())
			params[key] = value
	ALPHA = params["ALPHA"]
	LAMBDA = params["LAMBDA"]
	paramsFile.close()

	f = open(OUTPUT_W_FILE_NAME, "r")
	W = json.loads(f.readline().strip())
	f.close()

	f = open(OUTPUT_B_FILE_NAME, "r")
	b = float(f.readline().strip())
	f.close()
	numCorrect = 0.0
	numWrong = 0.0
	for i in range(0,NUM_INSTANCE_TO_PROCESS):
		instance = data[i]
		x = instance['vector']
		y = 0
		if class_number in instance['classes']:
			y = 1
		z = sparse_mult(W, x) + b
		a = sigma(z)
		predicted_label = 0
		if(a > simplePredictionA[i]):
			simplePredictionA[i] = a
			simplePredictionClass[i] = class_number
		if (a>= 0.5):
			predicted_label = 1
		if predicted_label == y:
			numCorrect += 1
		else:
			numWrong += 1
		if predicted_label == 1:
			if i in predicted_classes:
				predicted_classes[i].add(class_number)
			else:
				predicted_classes[i] = set([class_number])
	accuracy_percentage = 100.0*numCorrect/(numCorrect+numWrong)
	print("Class " + str(class_number) + " predictions done. Accuracy = " + "{0:.2f}".format(accuracy_percentage))

# individual class predictions. now we need to compare predicted set against true set of classes
print("All individual classes predictions done. Doing overall prediction...")

numWrong = 0.0
numCorrect = 0.0
numNotFound = 0
for i in range(0,NUM_INSTANCE_TO_PROCESS):
	if i not in predicted_classes:
		# print ("Instance " + str(i) + " not found in predicted_classes")
		numWrong += 1
		numNotFound += 1
		continue
	if(predicted_classes[i] == set(data[i]['classes'])):
		numCorrect += 1
	else:
		numWrong += 1

accuracy_percentage = 100.0*numCorrect/(numCorrect+numWrong)
print ("Exact Correct: " + str(numCorrect) + " Wrong: " + str(numWrong) + " Accuracy: " + "{0:.2f}".format(accuracy_percentage))
print ("Not found: " + str(numNotFound))

print("Doing simple prediction...")

numWrong = 0.0
numCorrect = 0.0
for i in range(0,NUM_INSTANCE_TO_PROCESS):
	if(simplePredictionClass[i] in data[i]['classes']):
		numCorrect += 1
	else:
		numWrong += 1
accuracy_percentage = 100.0*numCorrect/(numCorrect+numWrong)
print ("Simple Correct (At least 1 predicted is in true set): " + str(numCorrect) + " Wrong: " + str(numWrong) + " Accuracy: " + "{0:.2f}".format(accuracy_percentage))


