# Theano
import theano
from theano import tensor as T

# Other Libs
import os
import numpy
from PIL import Image, ImageOps
from six.moves import cPickle

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