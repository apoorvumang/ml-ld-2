#!/usr/bin/python
import re
import sys
import json
import numpy as np
import math

VOCAB_FILE_NAME = "vocab.txt"
CLASSES_FILE_NAME = "classes.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse.txt"
ALPHA = 0.01




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

count = 0
for line in file.readlines():
	line = json.loads(line)
	data.append(line)
	count = count + 1
	# if(count > 1000):
	# 	break

for epoch in range(0,5):
	J = 0
	for i in range(0,10000):
		instance = data[i]
		x = instance['vector']
		y = 0
		if 3 in instance['classes']:
			y = 1
		z = sparse_mult(W, x) + b
		a = sigma(z)
		dz = a - y
		sparse_dW_times_ALPHA = sparse_scalar_mult(x, dz*ALPHA)
		db = dz
		# actual loss
		J += -(y*math.log(a) + (1-y)*math.log(1 - a))
		# update parameters
		W = sparse_subtract(W, sparse_dW_times_ALPHA)
		b = b - ALPHA*db
	modW = np.dot(W,W)
	J = J/10000.0
	print ("Epoch done, loss = " + str(J))
	print ("Mod square of W = " + str(modW))



print(len(data))
file.close()