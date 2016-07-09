from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

import lc_helpers
import lc_logic


def react_frontend(request):
    return HttpResponseRedirect("/static/react_frontend/index.html")


@csrf_exempt
def process_image(request):
    try:
        key = request.FILES.keys()[0]
    except IndexError as e:
        print "Exception encountered:\n" + unicode(e)
        return JsonResponse({"error": unicode(e)})
    image = request.FILES[key]
    image_manager = lc_helpers.FileManager(image)
    path = image_manager.temp_path
    result = lc_logic.process(path)
    del image_manager
    return JsonResponse(result)