# Libs
import os
import tempfile
import magic
import numpy
import math


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
    step = total_points / number_of_points
    if index % number_of_points == 0:
        return True
    return False


def generate_chart_data(approximation=None, approximation_line=True):
    if approximation is None:
        PARAMETER = 5  # Hard coded for now, this will be guessed using theano
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
            return data
        else:
            data = []
            x_rand = numpy.linspace(-10, 10, 100)
            y_rand = approximation * x_rand
            for x, y in zip(x_rand, y_rand):
                data.append({"type": "scatter", "marker": {"color": "#000000"}, "x": [float(x)], "y": [float(y)]})
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
