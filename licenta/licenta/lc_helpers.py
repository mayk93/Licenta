# Libs
import os
import tempfile
import magic


TYPE_TO_EXT = {
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpg"
}


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
        print "[CHUNKED] Saving file here: " + path
        with open(path, "ab+") as destination:
            for chunk in self.managed_file.chunks():
                destination.write(chunk)
        return path

    def write_whole_filed(self):
        path = tempfile.mkstemp(prefix="whole_", suffix=self.file_extension)[1]
        with open(path, "ab+") as destination, self.managed_file as source:
                destination.write(source.read())
        return path


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
