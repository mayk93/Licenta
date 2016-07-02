from django.http import HttpResponseRedirect


def react_frontend(request):
    return HttpResponseRedirect("/static/index.html")