# Theano
import theano
from theano import tensor as T

# Other Libs
import numpy
from PIL import Image, ImageOps

# Django
from ..settings import PICKLED_OBJECTS_PATH

# Mine
from ..custom_decorators import custom


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

    multiplication_function = theano.function(inputs=[a, b], outputs=y)

    print " -> " + unicode(multiplication_function(10, 5))
    print " -> " + unicode(multiplication_function(4, 2))
    print " -> " + unicode(multiplication_function(5, 3))

    image = Image.open(file_path)
    image = ImageOps.grayscale(image)
    pixels_array = numpy.asarray(image)

    # pixels_object = image.load()
    # width, height = image.size
    #
    # pixels_array = []
    # for x in range(width):
    #     for y in range(height):
    #         pixel = pixels_object[x, y]
    #         pixels_array.append(pixel)

    print unicode(pixels_array)


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