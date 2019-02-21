
'''
Prob 1, part 2

Softmax being invariant to constants allows us to compare the relative probabilities of a class and not the magnitude

'''

# Problem 2 Iris dataset

# Part 1 - load data

import numpy as np
from numpy import loadtxt
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap

# part 2 - softmax classifier

class SoftmaxClassifier:

    def __init__(self, epochs, learning_rate, batch_size, regularization, momentum):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.momentum = momentum
        self.velocity = None
        self.weights = None

    def one_hot(self, y):
        # get a vector of labels, convert into 1 hot

        num_classes = 3  # needs to be fixed
        y = np.asarray(y, dtype='int32')  # convert type to int
        y = y.reshape(-1)  # convert into a list of numbers
        y_one_hot = np.zeros((len(y), num_classes))  # init shape of len y, and out 3 (num of classes)
        y_one_hot[np.arange(len(y)), y] = 1  # set the right indexes to 1, based on y (a list)
        return y_one_hot  # shape N by num_classes (3)

    def calc_accuracy(self, x, y):
        #  predict the class, then compare with the correct label.  return the average correct %
        pred = np.argmax(x.dot(self.weights), 1)  # predict
        pred = pred.reshape((-1, 1))  # convert to column vector
        return np.mean(np.equal(y, pred))  # return average over all the 1's (over the total)

    def softmax(self, x):
        # calc the softmax
        exp_x = np.exp(x - np.max(x))  # make sure it doesn't blow up by sub max

        # make sure sum along columns, and keep dims keeps the exact same dim when summing
        # ie keep cols, instead of converting to rows
        y = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / y

    def loss_and_gradient(self, x, y):
        # calc the loss and gradient.  forward prop, get softmax, calc the neg loss loss, and total loss.
        # calc dW by taking the residual, then dot with X,  + regularization
        # find average for both

        n_samples = x.shape[0]  # num of examples

        # forward prop
        f = np.dot(x, self.weights)  # mult X by W
        probs = self.softmax(f)  # pass f to softmax

        # take neg log of the highest prob. for that row
        neg_log_loss = -np.log(probs[np.arange(n_samples), np.argmax(probs, axis=1)])
        # neg_log_loss = -np.log(probs[np.arange(n_samples), y])
        loss = np.sum(neg_log_loss)  # sum to get total loss across all samples
        # calc the regularization loss too
        reg_loss = 0.5 * self.regularization * np.sum(self.weights * self.weights)
        total_loss = (loss / n_samples) + reg_loss  # sum to get total, divide for avg

        # calc dW
        y_one_hot = self.one_hot(y)  # need one hot

        # calc derivative of loss (including regularization derivative)
        dW = x.T.dot( (probs - y_one_hot) ) + (self.regularization * self.weights) 
        dW /= n_samples  # compute average dW

        return total_loss, dW

    def train_phase(self, x_train, y_train):
        # shuffle data together, and forward prop by batch size, and add momentum

        num_train = x_train.shape[0]
        losses = []
        # Randomize the data (using sklearn shuffle)
        x_train, y_train = shuffle(x_train, y_train)

        # get the next batch (loop through number of training samples, step by batch size)
        for i in range(0, num_train, self.batch_size):

            # grab the next batch size
            x_train_batch = x_train[i:i + self.batch_size]
            y_train_batch = y_train[i:i + self.batch_size]

            # forward prop
            loss, dW = self.loss_and_gradient(x_train_batch, y_train_batch)  # calc loss and dW
            # calc velocity
            self.velocity = (self.momentum * self.velocity) + (self.learning_rate * dW)
            self.weights -= self.velocity  # update the weights
            losses.append(loss)  # save the losses

        return np.average(losses)  # return the average

    def test_phase(self, x, y_test):
        # extra, but more explicit calc of loss and gradient during testing (no back prop)

        loss, _ = self.loss_and_gradient(x, y_test)  # calc loss and dW (don't need)
        return loss

    def run_epochs(self, x_train, y_train, x_test, y_test):
        # start the training/valid by looping through epochs

        num_dim = x_train.shape[1]  # num of dimensions
        n_classes = 3  # num output

        # create weights array/matrix size (num features x output)
        self.weights = 0.001 * np.random.rand(num_dim, n_classes)
        self.velocity = np.zeros(self.weights.shape)

        # store losses and accuracies here
        train_losses = []
        test_losses = []
        train_acc_arr = []
        test_acc_arr = []

        for e in range(self.epochs): # loop through epochs

            # print('Ephoch {} / {}...'.format(e + 1, self.epochs))

            # calc loss and accuracies
            train_loss = self.train_phase(x_train, y_train)
            test_loss = self.test_phase(x_test, y_test)
            train_acc = self.calc_accuracy(x_train, y_train)
            test_acc = self.calc_accuracy(x_test, y_test)

            # append vals to lists
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_acc_arr.append(train_acc)
            test_acc_arr.append(test_acc)

        return train_losses, test_losses, train_acc_arr, test_acc_arr  # return all the vals

            # print('Training loss {}'.format(train_loss))
            # print('Test loss {}'.format(test_loss))
            # print('Train Accuracy {}'.format(train_acc))
            # print('Test Accuracy {}'.format(test_acc))

    def plot_graph(self, train_losses, test_losses, train_acc, test_acc):
        # plot graph
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train loss")
        plt.plot(test_losses, label="Test loss")
        plt.legend(loc='best')
        plt.title("Epochs vs. Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss (Cross entropy)")

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label="Train Accuracy")
        plt.plot(test_acc, label="Test Accuracy")
        # plt.legend(loc='best')
        plt.title("Epochs vs Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.show()

    def make_mesh_grid(self, x, y, h=0.02):
        # make a mesh grid for the decision boundary
        
        x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        x_x, y_y = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return x_x, y_y  # matrix of x-axis and y-axis

    def plot_contours(self, plt, x_x, y_y, **params):
        # plot contours    

        array = np.array([x_x.ravel(), y_y.ravel()])
        f = np.dot(array.T, self.weights)
        prob = self.softmax(f)
        Q = np.argmax(prob, axis=1) + 1
        Q = Q.reshape(x_x.shape)
        plt.contourf(x_x, y_y, Q, **params)  # takes in variable number of params

    def plot_decision_boundary(self, x, y):
        # plot decision boundary

        markers = ('o', '.', 'x')
        colors = ('yellow', 'grey', 'green')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        x_x, y_y = self.make_mesh_grid(x, y)
        self.plot_contours(plt, x_x, y_y, cmap=plt.cm.coolwarm, alpha=0.8)
        
        # plot training points
        for idx, cl in enumerate(np.unique(y)):
            xBasedOnLabel = x[np.where(y[:,0] == cl)]
            plt.scatter(x=xBasedOnLabel[:, 0], y=xBasedOnLabel[:, 1], c=cmap(idx),
                        cmap=plt.cm.coolwarm, marker=markers[idx], label=cl)
        plt.xlim(x_x.min(), x_x.max())
        plt.ylim(y_y.min(), y_y.max())
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Decision Boundary - Softmax Classifier")
        plt.legend(loc='upper left')
        plt.show()


# load data
train_data = loadtxt('iris-train.txt')
x_train = train_data[:,1:]
y_train = train_data[:,0].astype(int)-1  # make sure to minus 1 for label
y_train = y_train.reshape((-1, 1))  # convert to column vector

test_data = loadtxt('iris-test.txt')
x_test = test_data[:,1:]
y_test = test_data[:,0].astype(int)-1  # make sure to minus 1 for label
y_test = y_test.reshape((-1, 1))   # convert to column vector


# set hyperparameters here
epochs = 1000
learning_rate = 0.01  # [0.1, 0.01, 0.001]
batch_size = 8  # try powers of 2
regularization = 0.01  # L2 weight decay, range [1, 0.1, 0.01, 0.001]
momentum = 0.10  # started with 0 to 1

smc = SoftmaxClassifier(epochs, learning_rate, batch_size, regularization, momentum)
train_losses, test_losses, train_acc, test_acc = smc.run_epochs(x_train, y_train, x_test, y_test)
smc.plot_graph(train_losses, test_losses, train_acc, test_acc)
smc.plot_decision_boundary(x_train, y_train)
smc.plot_decision_boundary(x_test, y_test)

# mean = 1











