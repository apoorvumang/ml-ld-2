#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math

VOCAB_FILE_NAME = "vocab.txt"
CLASSES_FILE_NAME = "classes.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse_test_full.txt"
W_FILE_NAME = "output_multi_w.txt"
B_FILE_NAME = "output_multi_b.txt"
# OUTPUT_W_FILE_NAME = "output_w.txt"
# OUTPUT_B_FILE_NAME = "output_b.txt"
# OUTPUT_PARAMS_FILE_NAME = "output_params.txt"
# OUTPUT_HISTORY_FILE_NAME = "output_history.txt"
ALPHA = 0.001
LAMBDA = 0.00


vocab = {}
f = open(VOCAB_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    word = splitLine[0]
    wordId = int(splitLine[1])
    vocab[word] = wordId
f.close()

classes = {}
f = open(CLASSES_FILE_NAME, "r")
for line in f.readlines():
    line = line.strip()
    splitLine = line.split('\t', 2)
    docClass = splitLine[0]
    classId = int(splitLine[1])
    classes[docClass] = classId
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
	if(z < -1000):
		print(z)
		return 0
	if(z > 1000):
		print(z)
		return 1
	return 1.0/(1.0 + math.exp(-z))

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


# read b and w

f = open("output_w.txt", "r")
W = json.loads(f.readline().strip())
f.close()

f = open("output_b.txt", "r")
b = json.loads(f.readline().strip())
f.close()

numCorrect = 0.0
numWrong = 0.0
for i in range(0,NUM_INSTANCE_TO_PROCESS):
	# huge and problematic instance
	# if (i == 121004):
	# 	continue
	instance = data[i]
	x = instance['vector']
	maxId = 0
	maxValue = 0.0
	for j in range(len(classes)):
		curr_W = W[j]
		curr_b = b[j]
		z = sparse_mult(curr_W, x) + curr_b
		a = sigma(z)
		if a > maxValue:
			maxValue = a
			maxId = j
	if maxId in instance['classes']:
		numCorrect += 1
	else:
		numWrong += 1

print ("Correct: " + str(numCorrect) + " Wrong: " + str(numWrong) + " Accuracy: " + str(numCorrect/(numCorrect+numWrong)))
print(len(data))