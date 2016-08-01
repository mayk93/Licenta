# Theano
import theano
from theano import tensor as T

# Other Libs
import numpy


'''
The probability that an input vector x is a member of a class i, a value of a stochastic variable Y, can be written as

<<< --- This may be wrong --- >>>
We have x, an n-dimensional vector. In the case of images, we can think of the pixels as being stringed together.

[[0, 0, 1],
 [1, 0, 1],
 [1, 1, 1]]

Say we had a 3 by 3 pixel image. If this were to be turned into our input vector x, it would be:
[0, 0, 1, 1, 0, 1, 1, 1, 1]. This vector represents the image we want to classify.
<<< --- >>>

i is the actual class. For simplicity, let's say i is binary, cat or not cat, 1 and 0. This is objective reality.

Y, a random (stochastic) variable is the prediction we generate. Ideally, Y should be i, but the prediction is sometimes
wrong. Y is not objective reality, Y is a prediction.

This is the formula for Y, given x.

P(Y=i|x,b,W) = softmax[i](Wx + b)

The probability ( P ) of Y being i ( the prediction matching objective reality ), conditioned by x (the input),
W(our weights) and b(a bias vector) is an activation function ( softmax[i] ) of the product of the input and weights plus
the bias vector. Input * Weights is essentially a neuron. The bias is the bias of that particular neuron.

Input layer                                    Layer 0

             w00 --- connects x0 to b0  ----- ( b0 )
          /                                      /
( x0 )  <-   w01 --- connects x0 to b1  -----   /
          \                                  \ /
             w02 --- connects x0 to b2  -     /
                                          \ / \
             w10 --- connects x1 to b0  ----- ( b1 )
          /                                \ |  |
( x1 )  <-   w11 --- connects x1 to b1      /  /
          \                                /\ /
             w12 --- connects x1 to b2 -\\/  /
                                         /\\/\
             w20 --- connects x2 to b0  -  /  \
          /                               /\\  \
( x2 )  <-   w21 --- connects x2 to b1  -   \\  \
          \                                  \\ |
             w22 --- connects x2 to b2  ----- ( b2 )

What we end up with in the second layer is something that is either activated ( true, is part of the class ) or not,
according to the softmax[i] function we choose.

Note! There are multiple softmax functions ( hence softmax[i] ), each belonging to a class.
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