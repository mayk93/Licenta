'''
Understanding linear classification:

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






