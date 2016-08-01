import json

from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

import lc_helpers

# Will be added - Now in the process of testing
import lc_logic

# Tests to learn Theano and Deep Learning
# from tests import opencv_tests, theano_tests, theano_tests_2

TEST_VERSION = 2


def react_frontend(request):
    return HttpResponseRedirect("/static/react_frontend/index.html")


@csrf_exempt
def get_chart_data(request):
    slope = int(json.loads(request.body).get("slope", 5))
    print "Got slope: " + unicode(slope)
    chart_data = lc_helpers.generate_chart_data(slope=slope)
    return JsonResponse({"data": chart_data})


@csrf_exempt
def approximate(request):
    data = json.loads(request.body)
    approximation = lc_logic.approximate(data)
    print "Approximation: " + unicode(approximation)
    print "Approximation Type: " + approximation.__class__.__name__
    approximation_chart_data = lc_helpers.generate_chart_data(approximation=approximation)
    return JsonResponse({"approximation": approximation_chart_data})


# @csrf_exempt
# def process_image_open_cv(request):
#     print "[Open CV] Received request"
#     image_manager = lc_helpers.file_request_setup(request)
#     if isinstance(image_manager, Exception):
#         print "[Open CV] Exception encountered:\n" + unicode(image_manager)
#         return JsonResponse({"error": unicode(image_manager)})
#     result = opencv_tests.process(image_manager.temp_path)
#     del image_manager
#     return JsonResponse(result)


# @csrf_exempt
# def process_image_open_theano(request):
#     print "[Theano] Received request"
#     image_manager = lc_helpers.file_request_setup(request)
#     if isinstance(image_manager, Exception):
#         print "[Theano] Exception encountered:\n" + unicode(image_manager)
#         return JsonResponse({"error": unicode(image_manager)})
#     # result = theano_tests.process(image_manager.temp_path)
#     if TEST_VERSION == 1:
#         result = theano_tests.process_digits(image_manager.temp_path)
#     else:
#         result = theano_tests_2.process_digits(image_manager.temp_path)
#
#     ###
#     '''
#     Here. we debug result for now.
#     '''
#     ###
#
#     print unicode(result)
#
#     del image_manager
#     return JsonResponse({"received": result})

@csrf_exempt
def process_image_open_theano(request):
    print "[Theano] Received request"
    image_manager = lc_helpers.file_request_setup(request)
    if isinstance(image_manager, Exception):
        print "[Theano] Exception encountered:\n" + unicode(image_manager)
        return JsonResponse({"error": unicode(image_manager)})

    result = lc_logic.process_digits(image_manager.temp_path)

    print unicode(result)

    del image_manager
    return JsonResponse({"prediction": result})