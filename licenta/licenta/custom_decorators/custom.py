import time


def time_decorator(function, *args):
    def wrapper(*args):
        start = time.time()
        function(*args)
        end = time.time()
        print "[Time Decorator] Execution took: " + unicode(end-start) + " seconds."
    return wrapper