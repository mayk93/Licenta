import time


def time_decorator(function):
    def wrapper():
        start = time.time()
        function()
        end = time.time()
        print "[Time Decorator] Execution took: " + unicode(end-start) + " seconds."
    return wrapper