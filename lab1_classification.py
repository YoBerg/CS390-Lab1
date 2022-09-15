from re import X
import sys
import os
import turtle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tensorflow as tf                                     # NOTE: This is just for utils. Do not use tensorflow in your code.
from tensorflow.keras.utils import to_categorical           # NOTE: This is just for utils. Do not use tensorflow in your code.
from torch.utils.data import TensorDataset, DataLoader
import json
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)
torch.manual_seed(1618)

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
# ALGORITHM = "custom_ann"
# ALGORITHM = "pytorch_ann"
ALGORITHM = "pytorch_cnn"

SAVE = True
LOAD = "model_in.pt"

# DATASET = "mnist_d"             # Handwritten digits.
DATASET = "mnist_f"            # Scans of types of clothes.
#DATASET = "cifar_10"           # Color images (10 classes).
#DATASET = "cifar_100_f"        # Color images (100 classes).
# DATASET = "cifar_100_c"        # Color images (20 classes).


'''
    In this lab, you will build a classifier system for several datasets and using several algorithms.
    Select your algorithm by setting the ALGORITHM constant.
    Select your dataset by setting the DATASET constant.

    Start by testing out the datasets with the guesser (already implemented).
    After that, I suggest starting with custom_ann on mnist_d.
    Once your custom_ann works on all datasets, try to implement pytorch_ann and pytorch_cnn.
    You must also add a confusion matrix and F1 score to evalResults.

    Feel free to change any part of this skeleton code.
'''



#==========================<Custom Neural Net>==================================

'''
    A neural net with 2 layers (one hidden, one output).
    Implement only with numpy, NO PYTORCH.
    A minibatch generator is already given.
'''

class Custom_ANN():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def _sigmoid(self, x):
        #TODO: implement
        return 1 / (1 + np.exp(-1 * x))

    # Activation prime function.
    def _sigmoidDerivative(self, x):    
        #TODO: implement    
        sig = self._sigmoid(x)
        return sig * (1 - sig)   

    # Batch generator for mini-batches. Not randomized.
    # **Appears to create generator breaking l into n sized segments.
    def _batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100, minibatches = True, mbs = 100):
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.

        # Run epoch
        for epoch in range(epochs):
            # Progress bar
            # sys.stdout.write('\r')
            # j = (epoch + 1) / epochs
            # sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
            sys.stdout.flush()

            # Create batches
            x_batches = self._batchGenerator(xVals, mbs)
            y_batches = self._batchGenerator(yVals, mbs)

            sse = 0

            # Run batches
            for x_batch, y_batch in zip(x_batches, y_batches):

                # Forward prop
                layer1, layer2 = self._forward(x_batch)

                error = np.sum((layer2 - y_batch) ** 2)
                sse += error

                # Back prop
                layer2_error = layer2 - y_batch
                layer2_delta = layer2_error * self._sigmoidDerivative(np.dot(layer1, self.W2))
                layer1_error = np.dot(layer2_delta, np.transpose(self.W2))
                layer1_delta = layer1_error * self._sigmoidDerivative(np.dot(x_batch, self.W1))
                layer1_adjustment = self.lr * (np.dot(np.transpose(x_batch), layer1_delta))
                layer2_adjustment = self.lr * (np.dot(np.transpose(layer1), layer2_delta))
                self.W1 -= layer1_adjustment
                self.W2 -= layer2_adjustment

            print("Epoch " + str(epoch) + "/" + str(epochs))
            sse /= len(yVals)
            print("Error: " + str(sse))
        sys.stdout.write('\n')
        # Done
        pass

    # Forward pass.
    def _forward(self, input):
        layer1 = self._sigmoid(np.dot(input, self.W1))
        layer2 = self._sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def __call__(self, xVals):
        # TODO: Reshaping x here, find better solution.
        _, layer2 = self._forward(xVals)

        # Make decision (set max to 1.0 and everything else to 0.0)
        # (Choose the classification we are most confident in)
        for i in range(len(layer2)):
            ind = 0
            max = 0
            for j in range(len(layer2[i])):
                if layer2[i][j] > max:
                    max = layer2[i][j]
                    ind = j
                layer2[i][j] = 0.0
            layer2[i][ind] = 1.0

        return layer2


#==========================<Pytorch Neural Net>=================================

'''
    A neural net built around nn.Linear.
    Use ReLU activations on hidden layers and Softmax at the end.
    You may use dropout or batch norm inside if you like.
'''

class Pytorch_ANN(nn.Module):
    def __init__(self):
        super(Pytorch_ANN, self).__init__()

        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Run forward propagation
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x




#==========================<Pytorch Convolutional Net>==========================

'''
    A neural net built around nn.Conv2d with one nn.Linear as the output layer.
    Use ReLU activations on hidden layers and Softmax at the end.
    You may use dropout or batch norm inside if you like.
'''

class Pytorch_CNN(nn.Module):
    def __init__(self):
        super(Pytorch_CNN, self).__init__()

        # First set of layers    [ Conv, Activation, Pool ]
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 24, kernel_size=(5,5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))

        self.dropout1 = nn.Dropout(p=0.3)

        # Second set of layers
        self.conv2 = nn.Conv2d(in_channels = 24, out_channels = 48, kernel_size=(5,5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))

        self.dropout2 = nn.Dropout(p=0.3)

        # Softmax classifier
        self.linear1 = nn.Linear(in_features = 48*4*4, out_features = 496)
        self.relu3 = nn.ReLU()

        self.dropout3 = nn.Dropout(p=0.3)

        self.linear2 = nn.Linear(in_features = 496, out_features = 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # Convert numpy array to torch tensor
        # x = torch.from_numpy(x)
        # # Permute tensor so we get [ Batch Channels Height Width]
        # print(x.size())
        # x = x[None, :]
        # print(x.size())
        # x = torch.permute(x, (1, 0, 2, 3))
        # print(x.size())
        # # Convert byte array to float array
        # x = x.float()

        # Pass through first set
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.dropout1(x)

        # Pass through second set
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.dropout2(x)

        # Pass through fully connected layer
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)

        x = self.dropout3(x)

        x = self.linear2(x)
        x = self.logsoftmax(x)

        return x

    def train_epochs(self, xTrain, yTrain, epochs = 100, lr = 0.001):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Create tensor dataset
        train_ds = TensorDataset(torch.from_numpy(xTrain).float(), torch.from_numpy(yTrain).float())
        train_dl = DataLoader(train_ds, batch_size=320)

        for epoch in range(epochs):
            print("Epoch # " + str(epoch))
            total_loss = 0
            for x_batch, y_batch in train_dl:
                # Ready for training
                self.train(True)       # Enable training
                optimizer.zero_grad()

                x_batch = x_batch.unsqueeze(1)
                # Run model 
                outputs = self(x_batch)    # Create predictions

                # Calculate loss
                # yTrain_tensor = torch.from_numpy(yTrain)
                loss = loss_fn(outputs, y_batch) 
                loss.backward()
                total_loss += loss

                # Adjust learning rates
                optimizer.step()
            # Report
            print("Loss: " + str(total_loss))
        return self



#===============================<Random Classifier>=============================

# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        numClasses = 10
        imgW, imgH, imgC = (28, 28, 1)
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
        numClasses = 10
        imgW, imgH, imgC = (28, 28, 1)
    elif DATASET == "cifar_10":
        cifar = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data()
        numClasses = 10
        imgW, imgH, imgC = (32, 32, 3)
    elif DATASET == "cifar_100_f":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="fine")
        numClasses = 100
        imgW, imgH, imgC = (32, 32, 3)
    elif DATASET == "cifar_100_c":
        cifar = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar.load_data(label_mode="coarse")
        numClasses = 20
        imgW, imgH, imgC = (32, 32, 3)
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return (((xTrain, yTrain), (xTest, yTest)), numClasses, imgW, imgH, imgC)



def preprocessData(raw, numClasses, imgW, imgH, imgC):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = xTrain.astype(float) / 255
    xTest = xTest.astype(float) / 255
    yTrainP = to_categorical(yTrain, numClasses)
    yTestP = to_categorical(yTest, numClasses)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))

    return ((xTrain, yTrainP), (xTest, yTestP))

def trainModel(data, numClasses, imgW, imgH, imgC, epochs = 100):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return guesserClassifier   # Guesser has no training, as it is just guessing.
    elif ALGORITHM == "custom_ann":
        xTrain = np.reshape(xTrain, (len(xTrain), 784))
        print("dimensions: " + str(imgW) + "," + str(imgH) + "," + str(imgC))
        print("Initializing with " + str(imgW * imgH * imgC) + " input size and " + str(numClasses) + " output size.")
        custom_ann = Custom_ANN(imgW * imgH * imgC, numClasses, 68, 0.01)

        custom_ann.train(xTrain, yTrain, epochs)
        return custom_ann
    elif ALGORITHM == "pytorch_cnn":
        model = Pytorch_CNN()
        model.train_epochs(xTrain, yTrain, epochs)
        return model
    elif ALGORITHM == "pytorch_ann":
        xTrain = np.reshape(xTrain, (len(xTrain), 784))
        model = Pytorch_ANN()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Create tensor dataset
        train_ds = TensorDataset(torch.from_numpy(xTrain).float(), torch.from_numpy(yTrain).float())
        train_dl = DataLoader(train_ds, batch_size=320)

        for epoch in range(epochs):
            print("Epoch # " + str(epoch))
            total_loss = 0
            for x_batch, y_batch in train_dl:

                # Prep for training
                model.train(True)       # This is enable training, not to start training.
                optimizer.zero_grad()
                outputs = model(x_batch) # Create predictions

                # Calculate Loss
                # yTrain_tensor = torch.from_numpy(yTrain)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                total_loss += loss

                # Adjust learning rates.
                optimizer.step()

            # Report
            print("Loss: " + str(total_loss))

        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your keras neural net.
        return None
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if (ALGORITHM == "custom_ann"):
        data = np.reshape(data, (len(data), 784))
        return model(data) 
    elif (ALGORITHM == "pytorch_ann"):
        data = np.reshape(data, (len(data), 784))
        tensor = torch.from_numpy(data).float()
        return model(tensor)
    else:
        tensor = torch.from_numpy(data).float()
        tensor = tensor.unsqueeze(1)
        return model(tensor)



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    # Catch if we need to cast tensors.
    if ALGORITHM == "pytorch_ann" or ALGORITHM == "pytorch_cnn":
        # Cast torch tensor to numpy array
        preds = preds.cpu().detach().numpy()
    
    # Make decision (set max to 1.0 and everything else to 0.0)
    # (Choose the classification we are most confident in)
    print("Pred before deciding: " + str(preds[0]))
    for i in range(len(preds)):
        ind = 0
        max = float("-inf")
        for j in range(len(preds[i])):
            if preds[i][j] > max:
                max = preds[i][j]
                ind = j
            preds[i][j] = 0.0
        preds[i][ind] = 1.0
    print("Pred after deciding: " + str(preds[0]))
    
    xTest, yTest = data
    acc = 0

    n = preds.shape[1]
    # Set up confusion matrix
    confusion_matrix = np.matrix(np.zeros((n, n)), int)

    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1

        # Compare true and predicted values for confusion matrix.
        pred = np.matrix(preds[i]).transpose()      # n by 1 matrix
        truth = np.matrix(yTest[i])                 # 1 by n matrix
        con = np.dot(pred, truth)                   # n by n matrix (only 1 value will be 1, others 0)
        confusion_matrix = confusion_matrix + con.astype(int)

    # Calculate accuracy.
    accuracy = acc / preds.shape[0]

    # Calculate totals.
    sum_vector = np.matrix(np.ones(n)) # 1 by n
    predicted_totals = np.dot(confusion_matrix, sum_vector.transpose())
    truth_totals = np.dot(sum_vector, confusion_matrix)
    total = np.dot(truth_totals, predicted_totals)

    # Print results.
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Confusion Matrix:    Columns add up to true values and rows add up to predictions")
    print(confusion_matrix)
    print("Truth Totals: ")
    print(truth_totals)
    print("Predicted Totals:") 
    print(predicted_totals.transpose())
    # print("Overall total: " + str(total[0,0]))        # This is wrong anyways.
    print()



#=========================<Main>================================================

def main():
    raw, nc, w, h, ch = getRawData()
    data = preprocessData(raw, nc, w, h, ch)
    model = None
    if LOAD is not None:
        print("LOAD found, Loading model from " + LOAD)
        if ALGORITHM == "pytorch_ann":
            model = Pytorch_ANN()
        else:
            model = Pytorch_CNN()
        model.load_state_dict(torch.load(LOAD))
        if False:
            xTrain, yTrain = data[0]
            model.train_epochs(xTrain,yTrain,30,0.0001)
    else:
        model = trainModel(data[0], nc, w, h, ch)
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)
    if SAVE and (ALGORITHM == "pytorch_ann" or ALGORITHM == "pytorch_cnn"):
        torch.save(model.state_dict(), "model_out.pt")



if __name__ == '__main__':
    main()
