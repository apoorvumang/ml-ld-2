import numpy as np
import ray
import time
import re
import sys
import json
import math
import sys
program_name = sys.argv[0]
arguments = sys.argv[1:]

class_number = 3
FULL = False
MAX_EPOCHS = 50
if len(arguments) == 3:
    class_number = int(arguments[0])
    if(arguments[1]=="full"):
        FULL = True
    MAX_EPOCHS = int(arguments[2])
else:
    print "Need 3 arguments: 1st argument is class number (0-49), 2nd is full/verysmall, 3rd is max number of epochs"
    exit()

CLASSES_FILE_NAME = "classes.txt"

VOCAB_FILE_NAME = "vocab_verysmall.txt"
DATA_VECTORS_FILE_NAME = "data/vectors_sparse_train_verysmall.txt"  
VALID_DATA_VECTORS_FILE_NAME = "data/vectors_sparse_valid_verysmall.txt"
if FULL:
    VOCAB_FILE_NAME = "vocab_full.txt"
    DATA_VECTORS_FILE_NAME = "data/vectors_sparse_train_full.txt"
    VALID_DATA_VECTORS_FILE_NAME = "data/vectors_sparse_valid_full.txt"
OUTPUT_W_FILE_NAME = "output_ps/output_w"+str(class_number)+".txt"
OUTPUT_B_FILE_NAME = "output_ps/output_b"+str(class_number)+".txt"
OUTPUT_PARAMS_FILE_NAME = "output_ps/output_params"+str(class_number)+".txt"
OUTPUT_HISTORY_FILE_NAME = "output_ps/output_history"+str(class_number)+".txt"
ALPHA = 0.01
LAMBDA = 0.00
NUM_WORKERS = 1




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
        print("Z value on overflow error was " + str(z))
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

vsigma = np.vectorize(sigma)

# sparse vector data with class





@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate):
        # Alternatively, params could be a dictionary mapping keys to arrays.
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
        self.W = np.zeros((VOCAB_SIZE,), dtype=np.float)
        self.b = 0.0

    def apply_gradients(self, *gradients):

        for grad in gradients:
            self.W = sparse_subtract(self.W, grad[0])
            self.b -= grad[1]
        return [self.W, self.b]

    def get_weights(self):
        return [self.W, self.b]

        
@ray.remote
class Worker(object):
    def __init__(self, worker_index, batch_size=50):
        self.current_instance = 0
        self.worker_index = worker_index
        self.batch_size = batch_size
        ALPHA = 0.005
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

        VOCAB_SIZE = len(vocab)

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

        chunkSize = NUM_INSTANCE_TO_PROCESS/NUM_WORKERS
        self.data = data[chunkSize*worker_index: chunkSize*worker_index + chunkSize]


    def compute_gradients(self, weights):
        W = weights[0]
        b = weights[1]
        data = self.data
        instance = data[self.current_instance]
        self.current_instance += 1
        if self.current_instance > len(data):
            self.current_instance = 0
        x = instance['vector']
        y = 0
        if class_number in instance['classes']:
            y = 1
        z = sparse_mult(W, x) + b
        a = sigma(z)
        dz = a - y
        sparse_dW_times_ALPHA = sparse_scalar_mult(x, dz*ALPHA)
        db = dz
        grad = [sparse_dW_times_ALPHA, ALPHA*db]
        return grad




ALPHA = 0.005
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

VOCAB_SIZE = len(vocab)

valid_data = []
file = open(VALID_DATA_VECTORS_FILE_NAME, "r")
NUM_INSTANCE_TO_PROCESS_VALID = 3000000 # if this exceeds no. of lines in file its still ok
count = 0
for line in file.readlines():
    line = json.loads(line)
    valid_data.append(line)
    count = count + 1
    if(count > NUM_INSTANCE_TO_PROCESS_VALID):
        break
NUM_INSTANCE_TO_PROCESS_VALID = count
file.close()

print ("class number is " + str(class_number))


if __name__ == "__main__":

    ray.init(redis_address=None)

    # Create a parameter server.
    ps = ParameterServer.remote(1e-4 * NUM_WORKERS)

    # Create workers.
    workers = [Worker.remote(worker_index)
               for worker_index in range(NUM_WORKERS)]


    epoch = 0
    current_weights = ps.get_weights.remote()
    output_history = []
    while True:
        # Compute and apply gradients.
        gradients = [worker.compute_gradients.remote(current_weights)
                     for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)

        if epoch % 10 == 0:

            # Evaluate the current model.
            # print(current_weights)
            weights = ray.get(current_weights)
            W = weights[0]
            b = weights[1]
            numCorrect = 0.0
            numWrong = 0.0
            J_VALID = 0
            for i in range(0, NUM_INSTANCE_TO_PROCESS_VALID):
                instance = valid_data[i]
                x = instance['vector']
                y = 0
                if class_number in instance['classes']:
                    y = 1
                z = sparse_mult(W, x) + b
                a = sigma(z)
                predicted = 0
                if a > 0.5:
                    predicted = 1
                if(y == predicted):
                    numCorrect+=1
                else:
                    numWrong+= 1
                if ((a<=0) or (a >= 1)):
                    continue
                J_VALID += -(y*math.log(a) + (1-y)*math.log(1 - a)) + LAMBDA*np.dot(W,W)
            J_VALID = J_VALID/NUM_INSTANCE_TO_PROCESS_VALID

            accuracy = numCorrect/(numWrong + numCorrect)
            print ("epoch "+ str(epoch) + ", validation loss: "+ str(J_VALID) + ", W square = " + str(np.dot(W,W)), " Accuracy " + str(accuracy))
            current_hist = {}
            current_hist['epoch'] = epoch
            current_hist['validation_loss'] = J_VALID
            current_hist['w_square'] = str(np.dot(W,W))
            output_history.append(current_hist)

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
            f.write("Validation Loss: " + str(J_VALID) + "\n")
            f.write("W squared: " + str(np.dot(W,W)) + "\n")
            f.close()

            f = open(OUTPUT_HISTORY_FILE_NAME, "w")
            f.write(json.dumps(output_history))
            f.close()
            # if J_VALID > J_VALID_PREVIOUS:
            #   print("Validation loss increased. Exiting...")
            #   break
            J_VALID_PREVIOUS = J_VALID

            
        epoch += 1

# output_history = []
# for epoch in range(100):
#     time.sleep(1)
#     x = ray.get([ps.get_params.remote() for ps in parameter_servers])
#     W = x[0][0]
#     b = x[0][1]


#     # validation data
    

#     numCorrect = 0.0
#     numWrong = 0.0
#     J_VALID = 0
#     for i in range(0, NUM_INSTANCE_TO_PROCESS_VALID):
#         instance = valid_data[i]
#         x = instance['vector']
#         y = 0
#         if class_number in instance['classes']:
#             y = 1
#         z = sparse_mult(W, x) + b
#         a = sigma(z)
#         predicted = 0
#         if a > 0.5:
#             predicted = 1
#         if(y == predicted):
#             numCorrect+=1
#         else:
#             numWrong+= 1
#         if ((a<=0) or (a >= 1)):
#             continue
#         J_VALID += -(y*math.log(a) + (1-y)*math.log(1 - a)) + LAMBDA*np.dot(W,W)
#     J_VALID = J_VALID/NUM_INSTANCE_TO_PROCESS_VALID

#     accuracy = numCorrect/(numWrong + numCorrect)
#     print ("Class "+ str(class_number) + " Epoch " + str(epoch) +  ", validation loss: "+ str(J_VALID) + ", W square = " + str(np.dot(W,W)), " Accuracy " + str(accuracy))
#     current_hist = {}
#     current_hist['epoch'] = epoch
#     current_hist['validation_loss'] = J_VALID
#     current_hist['w_square'] = str(np.dot(W,W))
#     output_history.append(current_hist)

#     f = open(OUTPUT_W_FILE_NAME, "w")
#     f.write(json.dumps(W.tolist()))
#     f.close()

#     f = open(OUTPUT_B_FILE_NAME, "w")
#     f.write(json.dumps(b))
#     f.close()

#     f = open(OUTPUT_PARAMS_FILE_NAME, "w")
#     f.write("Epochs done: " + str(epoch + 1) + "\n")
#     f.write("ALPHA: " + str(ALPHA) + "\n")
#     f.write("LAMBDA: " + str(LAMBDA) + "\n")
#     f.write("Validation Loss: " + str(J_VALID) + "\n")
#     f.write("W squared: " + str(np.dot(W,W)) + "\n")
#     f.close()

#     f = open(OUTPUT_HISTORY_FILE_NAME, "w")
#     f.write(json.dumps(output_history))
#     f.close()
#     # if J_VALID > J_VALID_PREVIOUS:
#     #   print("Validation loss increased. Exiting...")
#     #   break
#     J_VALID_PREVIOUS = J_VALID



