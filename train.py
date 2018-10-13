#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math

VOCAB_FILE_NAME = "vocab.txt"
CLASSES_FILE_NAME = "classes.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse.txt"
OUTPUT_W_FILE_NAME = "output_w.txt"
OUTPUT_B_FILE_NAME = "output_b.txt"
OUTPUT_PARAMS_FILE_NAME = "output_params.txt"
ALPHA = 0.01
LAMBDA = 0.00001


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

for epoch in range(0,100):
	J = 0
	for i in range(0,NUM_INSTANCE_TO_PROCESS):
		# huge and problematic instance
		if (i == 121004):
			continue
		instance = data[i]
		x = instance['vector']
		y = 0
		if 3 in instance['classes']:
			y = 1
		z = sparse_mult(W, x) + b
		a = sigma(z)

		if ((a<=0) or (a >= 1)):
			print("Instance no. " + str(i) + " caused issue")
			print(data[i]['vector'])
			continue
		dz = a - y
		sparse_dW_times_ALPHA = sparse_scalar_mult(x, dz*ALPHA)
		db = dz
		# actual loss
		J += -(y*math.log(a) + (1-y)*math.log(1 - a)) #+ LAMBDA*np.dot(W,W)
		# update parameters
		# need to add 2*LAMBDA*wi to each wi for l2 regularization?
		W = sparse_subtract(W, sparse_dW_times_ALPHA)
		# W = (1.0 - LAMBDA)*W 
		b = b - ALPHA*db
	J = J/NUM_INSTANCE_TO_PROCESS
	print ("Epoch " + str(epoch) + ", loss = " + str(J) + ", W square = " + str(np.dot(W,W)))
	f = open(OUTPUT_W_FILE_NAME, "w")
	f.write(json.dumps(W.tolist()))
	f.close()

	f = open(OUTPUT_B_FILE_NAME, "w")
	f.write(json.dumps(b))
	f.close()

	f = open(OUTPUT_PARAMS_FILE_NAME, "w")
	f.write("Epochs done: " + str(epoch + 1) + "\n")
	f.write("ALPHA: " + str(ALPHA) + "\n")
	f.write("LAMBDA: " + str(LAMBDA) + "\n")
	f.write("Number of instances: " + str(NUM_INSTANCE_TO_PROCESS) + "\n")
	f.write("Loss: " + str(J) + "\n")
	f.write("W squared: " + str(np.dot(W,W)) + "\n")
	f.close()
print(len(data))
file.close()
