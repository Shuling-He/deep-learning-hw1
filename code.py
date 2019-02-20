
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

    def one_hot(self, Y):
        # get a vector of labels, convert into 1 hot

        # get range of labels
        max_val = np.max(Y)
        min_val = np.min(Y)
        num_classes = int(max_val - min_val + 1)

        N = int(Y.shape[0])

        # create vector size N x num_classes
        one_hot_vector = np.zeros((N, num_classes), dtype=int)

        # fill one hot vector with values from Y, subtracting 1 to have 0 index
        one_hot_vector[np.arange(Y.shape[0]), Y.astype(int)-1] = 1

        return one_hot_vector 

    def softmax(self, X):

        exp_X = np.exp(X-np.max(X))

        # make sure sum along columns, and keepdims keeps the exact same dim when summing
        # ie keep cols, instead of converting to rows
        y = np.sum(exp_X, axis=1, keepdims=True)

        return exp_X / y

    def loss_and_gradient(self, X, Y, probs):

        '''
        X: numpy arr, shape = n_samples x 2 features
        Y: numpy arr, shape = n_samples x 1 label

        '''

        n_samples = X.shape[0]  # num of examples

        # take neg log of the prob. in the correct class (we want closest to 1)
        # inside probs, rows are 0-n_samples, cols are indexes in Y (labels)
        # then get the neg log of vector
        neg_log_loss = -np.log(probs[np.arange(n_samples), Y])

        # sum to get total loss across all samples
        loss = np.sum(neg_log_loss)

        one_hot_Y = self.one_hot(Y)
        dW = np.dot(X.T, (probs - one_hot_Y))

        # compute averages
        dW /= n_samples
        loss /= n_samples

        return loss, dW

    def train(self, train_X, train_Y, test_X, test_Y):

        # store losses and accuracies here
        train_losses = []
        test_losses = []
        # train_acc = []
        test_acc_arr = []

        # num of training samples
        num_train = train_X.shape[0]
        num_features = train_X.shape[1]

        n_classes = 3  # num output

        # Randomize the data (using sklearn shuffle)
        train_X, train_Y = shuffle(train_X, train_Y)

        # create W weight array/matrix size (num features x output). vals between 0 and 1
        W = np.random.rand(num_features, n_classes)

        velocity = np.zeros(W.shape)  # init vel with 0's and shape of W

        # loop through epochs
        for epoch in range(self.epochs):

            print('Ephoch {} / {}...'.format(epoch, self.epochs))

            # ============= Train phase ================= #

            running_train_loss = 0  # track loss
            
            # get the next batch (loop through number of training samples, step by batch size)
            for i in range(0, num_train, self.batch_size):

                train_X_batch = train_X[i:i + self.batch_size]
                train_Y_batch = train_Y[i:i + self.batch_size]

                # forward prop
                f = train_X.dot(W)  # mult X by W
                probs = self.softmax(f)  # pass f to softmax
                loss, dW = self.loss_and_gradient(train_X, train_Y, probs)  # calc loss and dW

                # calc regularization loss ( 0.5 * reg. * L2(W) )
                reg_loss = 0.5 * self.regularization * (W*W)
                loss += reg_loss  # add it

                # calc velocity (prev change)
                velocity = self.momentum * velocity + (learning_rate * dW)

                # update weights
                W -= velocity

                # append the loss
                running_train_loss += loss

            train_loss = running_train_loss / batch_size
            train_losses.append(train_loss)

            # =========  Test phase ================ #
            
            running_test_loss = 0  # validation phase
            running_total = 0  # running total
            running_correct = 0  # running correct

            # get the next batch (loop through number of training samples, step by batch size)
            for i in range(0, test_X.shape[0], self.batch_size):

                test_X_batch = test_X[i:i + self.batch_size]
                test_Y_batch = test_Y[i:i + self.batch_size]

                f = test_X.dot(W)  # mult X by W
                probs = self.softmax(f)  # pass f to softmax
                preds = np.argmax(probs, 1)  # returns the index of max values for each row (tells which column index is pred)
                preds = preds.reshape((-1, 1))  # convert to column vector
                running_total += self.batch_size  # add total size of batch
                running_correct += np.sum(np.equal(preds, test_Y_batch))  # sum corrects (note: not one hots)

                loss, _ = self.loss_and_gradient(test_X, test_Y, probs)  # calc loss (don't need dW) 
                running_test_loss += loss

            # calc the loss
            test_loss = running_test_loss / batch_size
            test_losses.append(test_loss)

            # calc the test accuracies
            test_acc = running_correct / running_total
            test_acc_arr.append(test_acc)

            print('Training loss {}'.format(train_loss))
            print('Test loss {}'.format(test_loss))
            print('Accuracy {}'.format(test_acc))


# load data
train_data = loadtxt('iris-train.txt')
train_X = train_data[:,1:]
train_Y = train_data[:,0].astype(int)-1  # make sure to minus 1 for label

test_data = loadtxt('iris-test.txt')
test_X = test_data[:,1:]
test_Y = test_data[:,0].astype(int)-1  # make sure to minus 1 for label



epochs = 100
learning_rate = 0.001  # [0.1, 0.01, 0.001]
batch_size = 8  # try powers of 2
regularization = 0.01  # L2 weight decay, range [1, 0.1, 0.01, 0.001]
momentum = 0.9  

smc = SoftmaxClassifier(epochs, learning_rate, batch_size, regularization, momentum)
smc.train(train_X, train_Y, test_X, test_Y)



# left
'''

- select hyper parameters
- incorporate weight decay


'''

















