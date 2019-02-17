
'''
Prob 1, part 2

Softmax being invariant to constants allows us to compare the relative probabilities of a class and not the magnitude

'''

# Problem 2 Iris dataset

# Part 1 - load data

import numpy as np
from numpy import loadtxt
from sklearn.utils import shuffle

# part 2 - softmax classifier

class SoftmaxClassifier:

    def __init__(self, epochs, learning_rate, batch_size, regularization, momentum):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.momentum = momentum
        self.velocity = None
        self.wt = None

    def one_hot(self, y):

        # get a vector of labels, convert into 1 hot

        num_classes = 3  # needs to be fixed

        y = np.asarray(y, dtype='int32')  # convert type to int
        y = y.reshape(-1)  # convert into a list of numbers

        y_one_hot = np.zeros((len(y), num_classes))
        y_one_hot[np.arange(len(y)), y] = 1
        return y_one_hot

    def predict(self, x):

        return np.argmax(x.dot(self.wt), 1)

    def mean_accuracy(self, x, y):
        """
        It calculates mean-per class accuracy
        :param x: Input sample
        :param y: label sample
        :return: mean-per class accuracy
        """
        
        pred = self.predict(x)
        pred = pred.reshape((-1, 1))  # convert to column vector

        return np.mean(np.equal(y, pred))

    def softmax(self, x):

        exp_x = np.exp(x - np.max(x))

        # make sure sum along columns, and keepdims keeps the exact same dim when summing
        # ie keep cols, instead of converting to rows
        y = np.sum(exp_x, axis=1, keepdims=True)

        return exp_x / y

    def loss_and_gradient(self, x, y):

        '''
    
        forward propagates, then calcs loss and gradient. Loss will includes regularization loss

        X: numpy arr, shape = n_samples x 2 features
        Y: numpy arr, shape = n_samples x 1 label

        '''

        n_samples = x.shape[0]  # num of examples

        # forward prop
        f = np.dot(x, self.wt)  # mult X by W
        probs = self.softmax(f)  # pass f to softmax

        # take neg log of the prob. in the correct class (we want closest to 1)
        # inside probs, rows are 0-n_samples, cols are indexes in Y (labels)
        # then get the neg log of vector
        neg_log_loss = -np.log(probs[np.arange(n_samples), y])
        loss = np.sum(neg_log_loss)  # sum to get total loss across all samples
        reg_loss = 0.5 * self.regularization * np.sum(self.wt * self.wt)

        print(loss)

        total_loss = (loss / n_samples) + reg_loss

        # calc dW
        y_one_hot = self.one_hot(y)  # need one hot

        # calc derivative of loss (including regularization derivative)
        dW = x.T.dot( (probs - y_one_hot) ) + (self.regularization * self.wt) 

        dW /= n_samples  # compute average dW

        return total_loss, dW

    def train_phase(self, x_train, y_train):

        num_train = x_train.shape[0]
        losses = []
        # Randomize the data (using sklearn shuffle)
        x_train, y_train = shuffle(x_train, y_train)

        # get the next batch (loop through number of training samples, step by batch size)
        for i in range(0, num_train, self.batch_size):

            x_train_batch = x_train[i:i + self.batch_size]
            y_train_batch = y_train[i:i + self.batch_size]

            # forward prop
            loss, dW = self.loss_and_gradient(x_train_batch, y_train_batch)  # calc loss and dW
            self.velocity = (self.momentum * self.velocity) + (self.learning_rate * dW)
            self.wt -= self.velocity
            losses.append(loss)

        return np.sum(losses) / len(losses)

    def test_phase(self, x, y_test):

        loss, _ = self.loss_and_gradient(x, y_test)  # calc loss and dW (don't need)
        return loss

    def run_epochs(self, x_train, y_train, x_test, y_test):

        num_dim = x_train.shape[1]  # num of dimensions
        n_classes = 3  # num output

        # create wt array/matrix size (num features x output)
        self.wt = 0.001 * np.random.rand(num_dim, n_classes)
        self.velocity = np.zeros(self.wt.shape)

        # store losses and accuracies here
        train_losses = []
        test_losses = []
        train_acc_arr = []
        test_acc_arr = []

        for e in range(self.epochs):

            print('Ephoch {} / {}...'.format(e + 1, self.epochs))

            train_loss = self.train_phase(x_train, y_train)
            test_loss = self.test_phase(x_test, y_test)
            train_acc = self.mean_accuracy(x_train, y_train)
            test_acc = self.mean_accuracy(x_test, y_test)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_acc_arr.append(train_acc)
            test_acc_arr.append(test_acc)

            print('Training loss {}'.format(train_loss))
            print('Test loss {}'.format(test_loss))
            print('Train Accuracy {}'.format(train_acc))
            print('Test Accuracy {}'.format(test_acc))

# load data

train_data = loadtxt('iris-train.txt')
x_train = train_data[:,1:]
y_train = train_data[:,0].astype(int)-1  # make sure to minus 1 for label
y_train = y_train.reshape((-1, 1))

test_data = loadtxt('iris-test.txt')
x_test = test_data[:,1:]
y_test = test_data[:,0].astype(int)-1  # make sure to minus 1 for label
y_test = y_test.reshape((-1, 1))


epochs = 1000
learning_rate = 0.07  # [0.1, 0.01, 0.001]
batch_size = 10  # try powers of 2
regularization = 0.001  # L2 weight decay, range [1, 0.1, 0.01, 0.001]
momentum = 0.05 

smc = SoftmaxClassifier(epochs, learning_rate, batch_size, regularization, momentum)
smc.run_epochs(x_train, y_train, x_test, y_test)



# left
'''

- select hyper parameters
- incorporate weight decay


'''

















