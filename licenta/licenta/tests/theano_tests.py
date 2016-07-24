# Theano
import theano
from theano import tensor as T

# Other Libs
import os
import numpy
from PIL import Image, ImageOps
from six.moves import cPickle

import gzip
import timeit

# Django
from ..settings import PICKLED_OBJECTS_PATH

# Mine
from ..custom_decorators import custom

'''
[

'''

@custom.time_decorator
def draft(file_path):
    '''
    Draft method for understanding theano. Used to learn. Will be deleted.
    :return:
    '''

    # These are 2 symbolic scalars. They are in a way types. We know that a and b are, in a sense, of type scalar.
    a = T.scalar()
    b = T.scalar()

    # This is a symbolic expression. We know that y is the product of a and b
    y = a * b

    # So far, this is all symbolic, it's an abstraction, nothing real here. We can't run this and get results. We need
    # to compile with theano.
    # This is the compiled function we can actually use

    if os.path.exists(PICKLED_OBJECTS_PATH + "multiplication_function.pk"):
        with open(PICKLED_OBJECTS_PATH + "multiplication_function.pk", "rb") as source:
            multiplication_function = cPickle.load(source)
    else:
        multiplication_function = theano.function(inputs=[a, b], outputs=y)
        with open(PICKLED_OBJECTS_PATH + "multiplication_function.pk", "w+") as destination:
            cPickle.dump(multiplication_function, destination, protocol=cPickle.HIGHEST_PROTOCOL)

    print " -> " + unicode(multiplication_function(10, 5))
    print " -> " + unicode(multiplication_function(4, 2))
    print " -> " + unicode(multiplication_function(5, 3))

    image = Image.open(file_path)
    image = ImageOps.grayscale(image)
    pixels_array = numpy.asarray(image)

    # Continue with Theano - Let's try a very simple matrix operation
    W = T.matrix("W")  # Weights
    x = T.matrix("x")  # Inputs
    dot = T.dot(x, W)  # Sum product ( value of "neuron" )
    y = T.nnet.sigmoid(dot)  # Activation function

    print unicode(pixels_array)


def process_data(raw_data):
    '''
    We know this is the structure we constructed for poltly in c_helpers. Now, we undo this structure.
    :param raw_data:
    :return:
    '''
    raw_list = raw_data["data"]
    x_values, y_values = [], []
    for item in raw_list:
        x_values.append(item["x"])
        y_values.append(item["y"])
    x_values = numpy.array(x_values)
    y_values = numpy.array(y_values)

    return x_values, y_values

# This is the actual symbolic model
def model(X, W):
    return X * W

def approximate(raw_data):
    # Important! These are nparrays
    x_values, y_values = process_data(raw_data)

    # Symbolic variables
    X = T.scalar()
    Y = T.scalar()

    # W is a hybrid variable. Shared variables need to have data associated with them at definition, but can also be
    # used in a symbolic context
    # We use W in out model as weights
    W = theano.shared(numpy.asarray(0., dtype=theano.config.floatX))
    # The result of applying the model. Used in computing the actual cost
    y = model(X, W)

    # The cost function
    # This tells us the 'error', how much 'off' we are in our prediction
    cost = T.mean(T.sqr(y - Y))
    # Using the grad function in theano, we get the partial derivative of the cost w.r.t ( with respect to ) the weights
    gradient = T.grad(cost=cost, wrt=W)
    # This is how we update the weights
    updates = [[W, W - gradient * 0.01]]

    # Compilation / load of model
    # allow_input_downcast is for GPU optimization - change 64 bit float to 32 bit float

    # Pickleing didn't go so well now - must change
    if os.path.exists(PICKLED_OBJECTS_PATH + "linear_model.pk"):
        with open(PICKLED_OBJECTS_PATH + "linear_model.pk", "rb") as source:
            linear_model = cPickle.load(source)
        linear_model = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    else:
        linear_model = theano.function(inputs=[X,Y], outputs=cost, updates=updates, allow_input_downcast=True)
        with open(PICKLED_OBJECTS_PATH + "linear_model.pk", "w+") as destination:
            cPickle.dump(linear_model, destination, protocol=cPickle.HIGHEST_PROTOCOL)

    for i in range(0, 100):
        for x, y in zip(x_values, y_values):
            linear_model(x[0], y[0])

    return W.get_value()


class TheanoImageProcessor(object):
    def __init__(self, image_path):
        self.image_path = image_path

    def __del__(self):
        pass

    def test(self):
        '''
        Silly test method that displays the image server side - Has no practical use
        :return:
        '''
        pass

def process(file_path):
    draft(file_path)

    image_processor = TheanoImageProcessor(file_path)
    image_processor.test()
    return {"path": file_path}



# ------



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
                      architecture (one minibatch)

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
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
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
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        # from six.moves import urllib
        # origin = (
        #     'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        # )
        raise Exception("[0] No dataset")
        # print('Downloading data from %s' % origin)
        # urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = cPickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

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

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset=PICKLED_OBJECTS_PATH + 'mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

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

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
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
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
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
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open(PICKLED_OBJECTS_PATH + 'best_model.pkl', 'wb') as f:
                        cPickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print ('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time)))


def predict(image):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    pixels_array = numpy.asarray(image).flatten()

    # print "To predict:\n" + unicode(pixels_array)
    #
    # dataset=PICKLED_OBJECTS_PATH + 'mnist.pkl.gz'
    # datasets = load_data(dataset)
    # test_set_x, test_set_y = datasets[2]
    # test_set_x = test_set_x.get_value()
    #
    # print "How it should look like:\n" + unicode(test_set_x[0])
    #
    # image.thumbnail((28, 28))
    #
    # pixels_array = numpy.asarray(image)
    #
    # print "After resize:\n" + unicode(pixels_array)


    # load the saved model
    classifier = cPickle.load(open(PICKLED_OBJECTS_PATH + 'best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    predicted_values = predict_model([pixels_array])[0]
    print("Predicted value:")
    print(predicted_values)
    return predicted_values


def process_digits(file_path, learn=False):
    if learn:
        sgd_optimization_mnist()
        return -1
    else:
        image = Image.open(file_path)

        predicted_value = predict(image)

        print "Predicated: " + unicode(predicted_value)

        return predicted_value