from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

import lc_helpers

# Will be added - Now in the process of testing
# import lc_logic

from tests import opencv_tests


def react_frontend(request):
    return HttpResponseRedirect("/static/react_frontend/index.html")


@csrf_exempt
def process_image_open_cv(request):
    print "[Open CV] Received request"
    try:
        key = request.FILES.keys()[0]
    except IndexError as e:
        print "Exception encountered:\n" + unicode(e)
        return JsonResponse({"error": unicode(e)})
    image = request.FILES[key]
    image_manager = lc_helpers.FileManager(image)
    path = image_manager.temp_path
    result = opencv_tests.process(path)
    del image_manager
    return JsonResponse(result)


@csrf_exempt
def process_image_open_theano(request):
    print "[Theano] Received request"
    return JsonResponse({"received": True})