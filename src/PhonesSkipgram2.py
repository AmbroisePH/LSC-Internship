# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:58:23 2016

@author: ambroise

GOAL: SKIPGRAM get phones around the targeted phone from the previous one

Here the n_out = 2*n_in. This output vector is cut in half, representing 
previous and next phone around the input phone. 
A loss function is calculated for each one of this half vector and added for 
the global loss function.

Input: outputs of GetPhones_dictio

Output: W model

This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5
                 
Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2
"""
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import codecs as cd

from gensim import corpora, models, similarities

import numpy
from pprint import pprint #pretty-printer

import theano
import theano.tensor as T

import fnmatch

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

########
##Load data, build dictionnary
########
#
#dictionary = corpora.Dictionary(line.lower().split() for line in cd.open('corpus1.txt', encoding='utf8'))
##dictionary = str(dictionary)
#
#N_OUT = len(dictionary)
#N_IN = N_OUT
#
#print(dictionary)
#print(dictionary.token2id)
#dico = dictionary.token2id
#
######## get the inputs ready
#
#
#with cd.open('corpus1.txt', 'r', encoding='utf8') as f:
#    text = [f.read().lower().split()]
#inp =  print(text)
#text=text[0]
#
#n_examples=len(text)
#
#train_set_x = numpy.zeros((n_examples,N_IN),dtype='int')
#
#
#for i in range (0,n_examples-1):
#    train_set_x[i,dico[text[i]]] = 1
#    
#print(train_set_x)

#
#train_set_y = numpy.zeros((n_examples,),dtype='int')
#
#train_set_y[n_examples-1]=numpy.nonzero(train_set_x[0])[0]
#for i in range (0,n_examples-2):
#    train_set_y[i]=numpy.nonzero(train_set_x[i+1])[0]
#
#print(train_set_y)

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      arcLogistichitecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k

        self.y_given_x = T.dot(input, self.W) + self.b # représentation recherchée ()     # 
        
        self.p_y_previous_given_x = T.nnet.softmax(self.y_given_x[:84/2-1])
        self.p_y_next_given_x = T.nnet.softmax(self.y_given_x[84/2:])
        

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_previous_pred = T.argmax(self.p_y_previous_given_x, axis=1)
        self.y_next_pred = T.argmax(self.p_y_next_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, ya, yb):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        
        
        NL_previous = -T.mean(T.log(self.p_y_previous_given_x)[T.arange(ya.shape[0]), ya])
        NL_next = -T.mean(T.log(self.p_y_previous_given_x)[T.arange(yb.shape[0]), yb])
        
        return NL_previous + NL_next        
        # end-snippet-2

    def errors(self, ya, yb):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        y_previous = ya
        y_next = yb
        
        # check if y has same dimension of y_pred
        if y_previous.ndim != self.y_previous_pred.ndim:
            raise TypeError(
                'y_previous should have the same shape as self.y_previous_pred',
                ('y_previous', y_previous.type, 'y_previous_pred', self.y_previous_pred.type)
            )
        
        
        # check if y has same dimension of y_pred
        if y_next.ndim != self.y_next_pred.ndim:
            raise TypeError(
                'y_next should have the same shape as self.y_next_pred',
                ('y_next', y_next.type, 'y_next_pred', self.y_next_pred.type,)
            )
        # check if y is of the correct datatype
        if y_next.dtype.startswith('int') and y_previous.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
        
            return T.mean(T.neq(self.y_previous_pred, y_previous)) + T.mean(T.neq(self.y_next_pred, y_next))
        else:
            raise NotImplementedError()




def load_data(dataset):
    
    #load buckeye dictionary 

    if fnmatch.fnmatchcase(dataset, '*real*'):
        dictio = corpora.Dictionary.load('BuckeyeDictionary_real.dict')
    elif fnmatch.fnmatchcase(dataset, '*dictio*'):
        dictio = corpora.Dictionary.load('BuckeyeDictionary_dictio.dict')
    else:
        raise TypeError('Filename does not contain real or dictio, load data cannot find its dictionary',(dataset))

    
    dico = dictio.token2id    
    print(dico)
    n_in = len(dictio)
    
    
    with cd.open(dataset, 'r', encoding='utf8') as f:
        text = f.read().lower().split()
  #      text=text[0]
        
    n_examples=len(text)
    print("n_examples = ", n_examples)
    
    train_set_x = numpy.zeros((n_examples,n_in),dtype='int')
    
    
    for i in range (0,n_examples-1):
        train_set_x[i,dico[text[i]]] = 1
    
    print("train_set_x")    
    print(train_set_x)
    
    
    train_set_y = [numpy.zeros((n_examples,),dtype='int'),numpy.zeros((n_examples,),dtype='int')]
    
    print(train_set_y)
   # train_set_y[0][n_examples-1]=numpy.nonzero(train_set_x[0])[0]
  #  train_set_y[0][0]=numpy.nonzero(train_set_x[n_examples-1])[0]
    for i in range (1,n_examples-1):
        train_set_y[0][i]=numpy.nonzero(train_set_x[i-1])[0]
        

   # train_set_y[1][n_examples-1]=numpy.nonzero(train_set_x[0])[0]
  #  train_set_y[1][0]=numpy.nonzero(train_set_x[n_examples-1])[0]
    for i in range (0,n_examples-2):
        train_set_y[1][i]=numpy.nonzero(train_set_x[i+1])[0]
        
      
    print("train_set_y")
    print(train_set_y)
    print(len(train_set_y[0]))
    print(len(train_set_x[4000:5000]))
    
    
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    arr = (train_set_x[:4000],[train_set_y[0][:4000],train_set_y[1][:4000]])
    train_set = tuple(map(tuple, arr))

    arr = (train_set_x[4000:5000],[train_set_y[0][4000:5000],train_set_y[1][4000:5000]])
    valid_set = tuple(map(tuple, arr))

    arr = (train_set_x[5000:],[train_set_y[0][5000:],train_set_y[1][5000:]])
    test_set = tuple(map(tuple, arr))    

#    print(test_set[1].eval().shape)
#    print(test_set[2].eval().shape)
    
    def shared_dataset(data_xy, borrow=True):
            """ Function that loads the dataset into             y_b: test_set_y[1][index * batch_size:(index + 1) * batch_size]shared variables
    
            The reason we store our dataset in shared variables is to allow
            Theano to copy it into the GPU memory (when code is run on GPU).
            Since copying data into the GPU is slow, copying a minibatch everytime
            is needed (the default behaviour if the data is not in a shared
            variable) would lead to a large decrease in performance.
            """
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            # When storing data on the GPU it has to be stored as floats
            # therefore we will store the labels as ``floatX`` as well
            # (``shared_y`` does exactly that). But during our computations
            # we need them as ints (we use labels as index, and if they are
            # floats it doesn't make sense) therefore instead of returning
            # ``shared_y`` we will have to cast it to int. This little hack
            # lets ous get around this issue
            return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(test_set)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
              (test_set_x, test_set_y)]
              
    print(train_set_x.eval().shape)
    print(train_set_y.eval().shape)
    print(valid_set_x.eval().shape)
    print(valid_set_y.eval().shape)
    print(test_set_x.eval().shape)
    print(test_set_y.eval().shape)
          
    return rval



# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.validate_model
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularizat10ion option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1, n_hidden=300):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    print(train_set_x.eval().shape)
    print(train_set_y.eval().shape)
    print(valid_set_x.eval().shape)
    print(valid_set_y.eval().shape)
    print(test_set_x.eval().shape)
    print(test_set_y.eval().shape)

    
    n_in = train_set_x.get_value().shape[1]
    print(n_in)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    ya = T.ivector('ya')
    yb = T.ivector('yb')                      # [int] labelstrain_set_x.get_value(borrow=True).shape[0]

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=2*n_in
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(ya, yb)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(ya,yb),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            ya: test_set_y[0][index * batch_size:(index + 1) * batch_size],
            yb: test_set_y[1][index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(ya,yb),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            ya: valid_set_y[0][index * batch_size:(index + 1) * batch_size],
            yb: valid_set_y[1][index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            ya: valid_set_y[0][index * batch_size:(index + 1) * batch_size],
            yb: valid_set_y[1][index * batch_size:(index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < validate_modelbest_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))



                    # save the best model
#                    with open('best_model_MLP1.pkl', 'wb') as f:
#                        for obj in [classifier.input,classifier.logRegressionLayer]:
#                            pickle.dump(obj, f)
#                        f.close()


#                    with open('best_model_MLP1_input.pkl', 'wb') as f:
#                         pickle.dump(classifier.input, f)
#                         
#                    with open('best_model_MLP1_LogRegressionLayer.pkl', 'wb') as f:
#                         pickle.dump(classifier.logRegressionLayer, f)
                         
                    with open('best_model_MLP1_params1.pkl', 'wb') as f:
                         pickle.dump(classifier.params, f)     
                         
#                    with open('best_model_MLP1_params2.pkl', 'wbwb') as f:
#                         pickle.dump(classifier.params, f)     
                         
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    os.chdir("/home/ambroise/Documents/LSC-Internship/data")
    test_mlp(learning_rate=0.1,n_epochs=1000,dataset="s3802a_dictio.words")

