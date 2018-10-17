#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math
import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]

class_number = 3
if len(arguments) == 1:
	class_number = int(arguments[0])

VOCAB_FILE_NAME = "vocab_full.txt"
CLASSES_FILE_NAME = "classes.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse_test_full.txt"
OUTPUT_W_FILE_NAME = "output_multi/output_w"+str(class_number)+".txt"
OUTPUT_B_FILE_NAME = "output_multi/output_b"+str(class_number)+".txt"
OUTPUT_PARAMS_FILE_NAME = "output_multi/output_params"+str(class_number)+".txt"
OUTPUT_HISTORY_FILE_NAME = "output_multi/output_history"+str(class_number)+".txt"

# OUTPUT_W_FILE_NAME = "output_w.txt"
# OUTPUT_B_FILE_NAME = "output_b.txt"
# OUTPUT_PARAMS_FILE_NAME = "output_params.txt"
# OUTPUT_HISTORY_FILE_NAME = "output_history.txt"

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


# read b and w

f = open(OUTPUT_W_FILE_NAME, "r")
W = json.loads(f.readline().strip())
f.close()

f = open(OUTPUT_B_FILE_NAME, "r")
b = float(f.readline().strip())
f.close()

numCorrect = 0.0
numWrong = 0.0
for i in range(0,NUM_INSTANCE_TO_PROCESS):
	# huge and problematic instance
	# if (i == 121004):
	# 	continue
	instance = data[i]
	x = instance['vector']
	y = 0
	if class_number in instance['classes']:
		y = 1
	z = sparse_mult(W, x) + b
	a = sigma(z)
	predicted_label = 0
	if (a>= 0.5):
		predicted_label = 1
	if predicted_label == y:
		numCorrect += 1
	else:
		numWrong += 1

print ("Class: "+str(class_number)+" Correct: " + str(numCorrect) + " Wrong: " + str(numWrong) + " Accuracy: " + str(numCorrect/(numCorrect+numWrong)))
print(len(data))
