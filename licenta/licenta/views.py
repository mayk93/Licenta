from django.http import HttpResponseRedirect


def react_frontend(request):
    return HttpResponseRedirect("/static/react_frontend/index.html")