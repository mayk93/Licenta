# Theano
import theano
from theano import tensor as T

# Other Libs
import os
import numpy

import dill as cPickle
from PIL import Image, ImageOps

# Django
from settings import PICKLED_OBJECTS_PATH

# This is for the simple slope approximation test

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
        linear_model = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)  # This is quite bad, I compute this every time. maybe use dill, that could work
    else:
        linear_model = theano.function(inputs=[X,Y], outputs=cost, updates=updates, allow_input_downcast=True)
        with open(PICKLED_OBJECTS_PATH + "linear_model.pk", "w+") as destination:
            cPickle.dump(linear_model, destination, protocol=cPickle.HIGHEST_PROTOCOL)

    for i in range(0, 100):
        for x, y in zip(x_values, y_values):
            linear_model(x[0], y[0])

    return W.get_value()

# -----  Here ends the simple slope approximation ----- #
# -----  This is where we begin the actual digit recognition ----- #


# from Learner import Learner  # This is our learning algorithm
from lc_helpers import get_test_data, get_training_data, get_validation_data


'''
Pickle and dill are stupid. I need to import the Learner class here to make this work.
'''

##### ===== ----- This is the learner class. This should be abstracted away, but abstracting breaks ----- ===== #####

'''
See learner class for more details. This was added here because of pickling issues.
'''

class Learner(object):
    '''
    This is a classifier. It is a classic regression classifier.
    We give it in input. The input is multiplied by a matrix of weights (W) and a bias is added.

    This works something like this:

    Inputs:

    ( x0 )  --- w00 ---  ( b0 )
            -         /
              \_ w01 --- ( b1 )
               \      | |
               |_w02 --- ( b2 )
                     / / /
    ( x1 )  --- w10 / / /
            \-- w11--/ /
            --- w12 --/

    ( x2 ) --- connects to b0, b1, b2 through w20, w21 and w22

    We model this as matrix multiplication.

    The input, x, is a matrix ( in Theano called Tensor ). We multiply the input by the weights and add the bias
    to the result.
    '''

    def __init__(self, input_matrix, input_size, output_size):
        '''

        :param input: The input matrix. This can be a vector, an image ( as a matrix ) or a section of an image, for DL
        :param input_size: The size of the input matrix. Needed to create appropriate sized weights.
        :param output_size: The size of the output matrix.
        '''

        # This is the input
        self.input = input_matrix

        # Here we initialize the weights
        self.W = theano.shared(value=numpy.zeros((input_size, output_size), dtype=theano.config.floatX),
                               name='W',
                               borrow=True)

        # Here we initialize the biases
        self.b = theano.shared(value=numpy.zeros((output_size,), dtype=theano.config.floatX),
                               name='b',
                               borrow=True)

        # Here we define the actual prediction probability. This is P(Y=i|x,b,W) = softmax[i](Wx + b)
        # Wx is modeled as a dot product, we add the bias and we apply the softmax function from the Theano library
        self.prediction_probability = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

        # Here we choose the most likely prediction.
        self.prediction = T.argmax(self.prediction_probability, axis=1)

        # The model has as parameters the weights and the bias
        self.params = [self.W, self.b]

    def cost_method(self, labels, negative_log=True):
        '''
        This is the cost method of the learning algorithm, as a function of the labels.
        When training the model, we use this function to penalize incorrect labeling.
        The function used is negative log of the probability.

        ===
        I'm still not very sure how this works. I don't understand why this works better than
        cost = T.mean(T.sqr(y - Y)), like the slope predictor.
        ===

        :param labels:
        :return:
        '''
        if negative_log:
            return -T.mean(T.log(self.prediction_probability)[T.arange(labels.shape[0]), labels])
        else:
            return T.mean(T.sqr(labels - self.prediction_probability))

    def errors(self, correct_labels):
        '''

        :param correct_labels: A vector with the correct labels. Used to compare with the predictions to compute an
                               error rate.
        :return: Error rate as a float. We know the correct labels, we know out prediction, we compute a ration between
                 the matches.
        '''

        return T.mean(T.neq(self.prediction, correct_labels))


class ClassificationModel(object):
    def __init__(self,
                 classifier,
                 test_model, validate_model, train_model,
                 no_of_train_batches, no_of_valid_batches, no_of_test_batches):

        self.classifier = classifier

        self.test_model = test_model
        self.validate_model = validate_model
        self.train_model = train_model

        self.no_of_train_batches = no_of_train_batches
        self.no_of_valid_batches = no_of_valid_batches
        self.no_of_test_batches = no_of_test_batches


class IterationResult(object):
    def __init__(self,
                 classification_model,
                 iteration_number,
                 validation_frequency,
                 best_validation_loss,
                 min_no_of_examples,
                 example_increase,
                 improvement_threshold,
                 stop_iterating):

        self.classification_model = classification_model
        self.iteration_number = iteration_number
        self.validation_frequency = validation_frequency
        self.best_validation_loss = best_validation_loss
        self.min_no_of_examples = min_no_of_examples
        self.example_increase = example_increase
        self.improvement_threshold = improvement_threshold
        self.stop_iterating = stop_iterating


##### ===== ----- These clases should have been abstracted away ----- ===== #####


def build_model(dataset_path, batch_size, learning_rate):
    '''

    See gradient_descent for argument datails.

    :param dataset_path:
    :param batch_size:
    :param learning_rate:

    :return: We implement a separation of concerns between building the model and training it. Here, we construct the
             necessary theano compiled functions and package them in an object to be used in training.
    '''
    # Get the data. We use the helper methods to unpickle the MINDS dataset and get the 3 data sets
    train_set_x, train_set_y = get_training_data(dataset_path=dataset_path)
    valid_set_x, valid_set_y = get_validation_data(dataset_path=dataset_path)
    test_set_x, test_set_y = get_test_data(dataset_path=dataset_path)

    # Instantiate the classification algorithm
    # We know the input are MINST images of digits that are 28 by 28 pixels in size. So input size is 28 * 28.
    # We know there are 10 digits, so the output size is 10.

    # The current batch we are indexing
    index = T.lscalar()

    # Symbolically declare the image ( input ) and the labels ( output )
    # The image is a matrix
    image_matrix = T.matrix('image_matrix')
    # The labels are a vector
    digit_labels = T.ivector('digit_labels')
    classifier = Learner(input_matrix=image_matrix, input_size=28 * 28, output_size=10)

    # Compute the cost using the learners cost method
    cost = classifier.cost_method(digit_labels)

    # Using this compiled function we test the model
    '''
    What this is saying:

    Take the images from the current point ( index ) until the end of the batch ( so, get the current batch from the
    test set ) and compute the error rate.

    If index = 10 and batch size is 10, this would be this batch:

    Test set: [ [image_0] [image_1] [image_2] ... [image_index_10] ... [image_20] ... [image_n] ]
                                                  ^                             ^
                                                Start batch here               End batch here
    '''
    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(digit_labels),
                                 givens={
                                     image_matrix: test_set_x[index * batch_size: (index + 1) * batch_size],
                                     digit_labels: test_set_y[index * batch_size: (index + 1) * batch_size]
                                 })

    # Using this compiled function we test the model
    validate_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(digit_labels),
                                     givens={
                                         image_matrix: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                         digit_labels: valid_set_y[index * batch_size: (index + 1) * batch_size]
                                     })

    # The learning method we are applying is gradient descent. So, we compute the gradient with respect to the weights
    # and the bias.
    # Here, wrt stands for with respect to. Theanos built in grad function allows us to easily compute the partial
    # derivatives of a matrix.
    weights_gradient = T.grad(cost=cost, wrt=classifier.W)
    bias_gradient = T.grad(cost=cost, wrt=classifier.b)

    # The model learns by updating it's self. Knowing the gradient, we must now specify how to apply the updates.
    # We generate a symbolic list of (value, update) pairs that is propagates in the model.
    updates = [(classifier.W, classifier.W - learning_rate * weights_gradient),
               (classifier.b, classifier.b - learning_rate * bias_gradient)]

    # Iterate over the batches and update the model. This is compiled method that will be invoked during training.
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={
                                      image_matrix: train_set_x[index * batch_size: (index + 1) * batch_size],
                                      digit_labels: train_set_y[index * batch_size: (index + 1) * batch_size]
                                  })

    # Here we compute the number of batches used for training, validation and testing
    no_of_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    no_of_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    no_of_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    return ClassificationModel(classifier,
                               train_model, validate_model, test_model,
                               no_of_train_batches, no_of_valid_batches, no_of_test_batches)


def train(classification_model,
          iteration_number,
          validation_frequency,
          best_validation_loss,
          min_no_of_examples,
          example_increase,
          improvement_threshold,
          stop_iterating):

    for batch_index in range(classification_model.no_of_train_batches):

        classification_model.train_model(batch_index)
        current_iteration = (iteration_number - 1) * classification_model.no_of_train_batches + batch_index

        # If it's the case for validation
        if (current_iteration + 1) % validation_frequency == 0:
            validation_losses = [classification_model.validate_model(i)
                                 for i in range(classification_model.no_of_valid_batches)]
            this_validation_loss = numpy.mean(validation_losses)

            # Check if we minimized the loss
            if this_validation_loss < best_validation_loss:
                # Keep iterating if an improvement has occurred
                if this_validation_loss < (best_validation_loss * improvement_threshold):
                    min_no_of_examples = max(min_no_of_examples, current_iteration * example_increase)

                best_validation_loss = this_validation_loss

                # Also test on the test set
                test_losses = [classification_model.test_model(i)
                               for i in range(classification_model.no_of_test_batches)]

        if current_iteration > min_no_of_examples:
            stop_iterating = True
            break

    return IterationResult(classification_model,
                           iteration_number,
                           validation_frequency,
                           best_validation_loss,
                           min_no_of_examples,
                           example_increase,
                           improvement_threshold,
                           stop_iterating)

def train_model(classification_model,
                no_of_iterations,
                min_no_of_examples=5000,
                example_increase=2,
                improvement_threshold=0.995):
    '''

    :param classification_model: The previously built classification model.
    :param no_of_iterations: Number of iterations over the training set.
    :param min_no_of_examples: This is the minimum number of MINST digits we will use to learn.
    :param example_increase: When an improvement occurred, increase the number of examples.
    :param improvement_threshold: If the error rate improves by this, we consider this a significant improvement.
    :return:
    '''

    '''
    This particular approach to training involves iterating several ( no_of_iterations ) times over the training set.
    However, by looking at the error rate, we can determine if we can stop early. We do this if there is no major
    improvement.
    '''

    # This is how often we test the model (how well it has been trained so far) on the validation samples.
    validation_frequency = min(classification_model.no_of_train_batches, min_no_of_examples / 2)

    # Keep track of error rate and score. At first, loss is "infinity" as we seek to minimize it.
    best_validation_loss = numpy.inf

    # Keep track of the iteration number and if we should stop iterating.
    stop_iterating = False
    iteration_number = 0

    while True:
        result = train(classification_model,
                       iteration_number,
                       validation_frequency,
                       best_validation_loss,
                       min_no_of_examples,
                       example_increase,
                       improvement_threshold,
                       stop_iterating)

        classification_model = result.classification_model
        iteration_number = result.iteration_number
        validation_frequency = result.validation_frequency
        best_validation_loss = result.best_validation_loss
        min_no_of_examples = result.min_no_of_examples
        example_increase = result.example_increase
        improvement_threshold = result.improvement_threshold
        stop_iterating = result.stop_iterating

        print "At iteration " + \
              unicode(iteration_number) + \
              " with best validation loss: " + \
              unicode(best_validation_loss)

        iteration_number = iteration_number + 1
        if iteration_number > no_of_iterations or stop_iterating:
            break

    return classification_model.classifier


def gradient_descent(dataset_path=PICKLED_OBJECTS_PATH + 'mnist.pkl.gz',
                     batch_size=600,
                     learning_rate=0.15,
                     no_of_iterations=1000):

    '''
    This is an implementation of the gradient decent learning method, using Theano.
    With the MINST data set, we do this:

    1. Build a classification model.
    2. Train the classification model.
    3. Save the trained classifier as a pickled object for future use.

    :param dataset_path: A path to the MINST dataset as a pickled object
    :param batch_size: Training occurs in batches. Look at batch_size images during one iteration.
    :param learning_rate:

    Gradient descent tries to approximate a local minimum or maximum of a function.
    It does this by descending towards that point at every iteration.
    The learning rate is the size of the step we make at every descent.
    It is tempting to make the learning rate big, but this increases the likelihood we go past our target.
    This is why the learning rate should be a small value.

    :param no_of_iterations: Since we learn in batches, we can iterate with "knowledge" gained from batch 2 for example
                             over batch 1, and ensure we keep improving. Essentially, we iterate multiple times over the
                             training set.
    :return:
    '''

    print "Now starting gradient Descent."

    classification_model = build_model(dataset_path, batch_size, learning_rate)

    print "Built the model. Now training the model."

    classifier = train_model(classification_model, no_of_iterations)

    print "Trained model. Now saving model as minst_digit_classifier."

    with open(PICKLED_OBJECTS_PATH + 'minst_digit_classifier.pkl', 'wb') as destination:
        cPickle.dump(classifier, destination)

    print "Saved the classifier. Gradient descent now done."


def predict(image):
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    pixels_array = numpy.asarray(image).flatten()

    # classifier = cPickle.load(open(PICKLED_OBJECTS_PATH + 'best_model.pkl'))
    classifier = cPickle.load(open(PICKLED_OBJECTS_PATH + 'minst_digit_classifier.pkl'))

    # Compile a predictor function
    predict_model = theano.function(inputs=[classifier.input],
                                    outputs=classifier.prediction)

    predicted_value = predict_model([pixels_array])[0]
    return predicted_value


def process_digits(file_path, learn=False):
    if learn:
        gradient_descent()
        return -1
    else:
        image = Image.open(file_path)

        predicted_value = predict(image)

        return predicted_value


if __name__ == '__main__':
    gradient_descent()