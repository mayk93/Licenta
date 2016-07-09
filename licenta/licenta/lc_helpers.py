# Libs
import os
import tempfile


TYPE_TO_EXT = {
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpg"
}


def extension_from_type(file_type):
    return TYPE_TO_EXT.get(file_type, "")


class FileManager(object):
    def __init__(self, managed_file, chunked=True):
        self.managed_file = managed_file
        self.file_extension = extension_from_type(self.managed_file.content_type)
        if chunked:
            self.temp_path = self.write_chunked_filed()
        else:
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
        path = tempfile.mkstemp(prefix="whole_", suffix=self.file_extension)
        with open(path, "ab+") as destination, self.managed_file as source:
                destination.write(source.read())
        return path