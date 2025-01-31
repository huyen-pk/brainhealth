from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from .forms import ImageUploadForm

def home(request):
    context = {
        'app_name': settings.APP_NAME
    }
    if request.method == "POST" and request.FILES.get("image"):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            # analyze image and send response to the client
            return JsonResponse({"message": "Image uploaded successfully!", "image_url": uploaded_image.image.url})
        else:
            return JsonResponse({"error": "Invalid file format"}, status=400)

    # form = ImageUploadForm()
    return render(request, "home/index.html", context)

