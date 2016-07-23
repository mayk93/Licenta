import theano
from theano import tensor as T

def draft():
    '''
    Draft method for understanding theano. Used to learn. Will be deleted.
    :return:
    '''
    pass


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
    draft()

    image_processor = TheanoImageProcessor(file_path)
    image_processor.test()
    return {"path": file_path}