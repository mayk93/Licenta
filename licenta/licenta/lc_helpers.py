# Libs
import os
import tempfile
import magic
import numpy
import math
import gzip
import theano
from theano import tensor as T
from six.moves import cPickle


TYPE_TO_EXT = {
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpg"
}

'''
          {
            type: 'scatter',  // all "scatter" attributes: https://plot.ly/javascript/reference/#scatter
            x: [1],     // more about "x": #scatter-x
            y: [2],     // #scatter-y
            marker: {         // marker is an object, valid marker keys: #scatter-marker
            }
          },
          {
            type: 'scatter',  // all "scatter" attributes: https://plot.ly/javascript/reference/#scatter
            x: [3],     // more about "x": #scatter-x
            y: [4],     // #scatter-y
            marker: {         // marker is an object, valid marker keys: #scatter-marker
            }
          }
        ];
'''

'''
0     5    10    15    25
|-----|-----|-----|-----|

25(total) / 5(nop) = 5(step)

index % step = 0?

'''


def is_selection_point(index, number_of_points, total_points):
    if number_of_points == 0:
        print "Called with number of points 0. Defaulting to 1"
        number_of_points = 1
    step = total_points / number_of_points
    if index % number_of_points == 0:
        return True
    return False


def generate_chart_data(approximation=None, approximation_line=True, slope=5):
    if approximation is None:
        PARAMETER = slope
        data = []
        x_rand = numpy.linspace(-10, 10, 100)
        y_rand = PARAMETER * x_rand + numpy.random.randn(*x_rand.shape) * 5
        for x, y in zip(x_rand, y_rand):
            data.append({"type": "scatter", "x": [float(x)], "y": [float(y)]})
        return data
    else:
        if approximation_line:
            data = []
            x_rand = numpy.linspace(-10, 10, 100)
            y_rand = approximation * x_rand
            x_array = []
            y_array = []
            number_of_points = math.floor(approximation)
            for index, (x, y) in enumerate(zip(x_rand, y_rand)):
                if is_selection_point(index, number_of_points, len(x_array)):
                    x_array.append(float(x))
                    y_array.append(float(y))
            data.append({"type": "scatter", "marker": {"color": "#000000"}, "x": x_array, "y": y_array})
            print "Returning Approximation plot."
            return data
        else:
            data = []
            x_rand = numpy.linspace(-10, 10, 100)
            y_rand = approximation * x_rand
            for x, y in zip(x_rand, y_rand):
                data.append({"type": "scatter", "x": [float(x)], "y": [float(y)]}) # , "marker": {"color": "#000000"}
            print "Returning Scatter plot."
            return data


def extension_from_type(file_type):
    return TYPE_TO_EXT.get(file_type, "")


class FileManager(object):
    def __init__(self, managed_file, chunked=True):
        self.managed_file = managed_file
        if chunked:
            self.file_extension = extension_from_type(self.managed_file.content_type)
            self.temp_path = self.write_chunked_filed()
        else:
            magic_type_detector = magic.Magic(mime=True, uncompress=True)
            self.file_extension = extension_from_type(magic_type_detector.from_buffer(self.managed_file.read(1024)))
            self.temp_path = self.write_whole_filed()

    def __del__(self):
        os.remove(self.temp_path)

    def write_chunked_filed(self):
        path = tempfile.mkstemp(prefix="chunked_", suffix=self.file_extension)[1]
        with open(path, "ab+") as destination:
            for chunk in self.managed_file.chunks():
                destination.write(chunk)
        return path

    def write_whole_filed(self):
        path = tempfile.mkstemp(prefix="whole_", suffix=self.file_extension)[1]
        with open(path, "ab+") as destination, self.managed_file as source:
                destination.write(source.read())
        return path


def file_request_setup(request):
    '''
    This method takes a request and sets up a file manager
    :param request: A request with a file
    :return: A FileManager object
    '''
    try:
        key = request.FILES.keys()[0]
    except IndexError as e:
        return e

    managed_file = request.FILES[key]
    file_manager = FileManager(managed_file)

    return file_manager


'''
Important!

These 3 methods have a default behaviour of shared. Because Theano is optimized to run on GPUs, we can load data into
the GPU memory. This increases performance. Since we don't need the data sets to be loaded every time, we can just put
them in a shared variable. It would be very slow to load them all the time. Turning off shared will return the raw data
set, as read from the pickled source. Performance will decrease.
'''


def split_and_share(data_set):
    '''

    :param data_set: A data set, read from the pickled object
    :return: 2 theano shared objects, one for the x tensor and another for the y tensor
    '''

    data_x, data_y = data_set

    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX),
                             borrow=True)

    return shared_x, T.cast(shared_y, 'int32')


def get_training_data(dataset_path, shared=True):
    try:
        with gzip.open(dataset_path, 'rb') as source:
            try:
                train_set, _, _ = cPickle.load(source, encoding='latin1')
            except:
                train_set, _, _ = cPickle.load(source)
            if not shared:
                print "[get_training_data] Returning raw data. This is bad and may break stuff."
                return train_set
            else:
                train_set_x, train_set_y = split_and_share(train_set)
    except IOError:
        print "[get_training_data] No data found."
        return None
    return train_set_x, train_set_y


def get_validation_data(dataset_path, shared=True):
    try:
        with gzip.open(dataset_path, 'rb') as source:
            try:
                _, validation_set, _ = cPickle.load(source, encoding='latin1')
            except:
                _, validation_set, _ = cPickle.load(source)
            if not shared:
                print "[get_validation_data] Returning raw data. This is bad and may break stuff."
                return validation_set
            else:
                validation_set_x, validation_set_y = split_and_share(validation_set)
    except IOError:
        print "[get_validation_data] No data found."
        return None
    return validation_set_x, validation_set_y


def get_test_data(dataset_path, shared=True):
    try:
        with gzip.open(dataset_path, 'rb') as source:
            try:
                _, _, test_set = cPickle.load(source, encoding='latin1')
            except:
                _, _, test_set = cPickle.load(source)
            if not shared:
                print "[get_test_data] Returning raw data. This is bad and may break stuff."
                return test_set
            else:
                test_set_x, test_set_y = split_and_share(test_set)
    except IOError:
        print "[get_test_data] No data found."
        return None
    return test_set_x, test_set_y


import unittest


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.t_path = "/var/folders/pb/tbyb3w3n7g34pqcvby4m8sj40000gn/T/"

    def test_extension_from_type(self):
        type_jpeg = "image/jpeg"
        self.assertEquals(".jpeg", extension_from_type(type_jpeg))

        self.assertEquals("", extension_from_type("text/html"))

    def test_file_manager(self):
        # Test content
        content = "This is the test content"
        # Create a file somewhere
        with open("/Users/Michael/Desktop/test_file.txt", "w+") as test_file:
            # Write some content
            test_file.write(content)
        # Get a list of the current files in the temp directory
        pre_existing_temp_files = [tf for tf in os.listdir(self.t_path) if os.path.isfile(os.path.join(self.t_path,
                                                                                                       tf))]
        print "Pre existing files: " + unicode(pre_existing_temp_files)
        # Now use the file manager to create a temp file with the same contents
        with open("/Users/Michael/Desktop/test_file.txt", "rb") as test_file:
            file_manager = FileManager(test_file, chunked=False)
            path = file_manager.temp_path
            temp_file = path.split("/")[len(path.split("/")) - 1]

            print "Created path: " + path
            print "Created file: " + temp_file

            # Now, assert the path was not among the pre existing files
            self.assertNotIn(temp_file, pre_existing_temp_files)

            # Get temp files again
            current_temp_files = [tf for tf in os.listdir(self.t_path) if os.path.isfile(os.path.join(self.t_path,
                                                                                                      tf))]

            # Make sure the file is in the current ones
            self.assertIn(temp_file, current_temp_files)

            # Now delete the file manager
            del file_manager

            # Now make sure the file disappeared
            current_temp_files = [tf for tf in os.listdir(self.t_path) if os.path.isfile(os.path.join(self.t_path,
                                                                                                      tf))]
            self.assertNotIn(temp_file, current_temp_files)

        # Remove the test file
        os.remove("/Users/Michael/Desktop/test_file.txt")
