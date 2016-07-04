from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt


def react_frontend(request):
    return HttpResponseRedirect("/static/react_frontend/index.html")

@csrf_exempt
def process_image(request):
    key = request.FILES.keys()[0]
    image = request.FILES[key]
    with open("/Users/Michael/Desktop/uploaded_cat.jpg", "ab+") as destination:
        for chunk in image.chunks():
            destination.write(chunk)
    return HttpResponse("Ok")