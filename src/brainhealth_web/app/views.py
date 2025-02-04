from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
import requests
from .forms import FileUploadForm
import tensorflow as tf
import os
from .utilities import format_image
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout
from .forms import UserRegistrationForm

@login_required
def home(request):
    context = {
        'app_name': settings.APP_NAME
    }
    
    if request.method == "POST":
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Read image content from file with OpenCV
            image_files = request.FILES.getlist("files")
            batch = []
            for image_file in image_files:
                image_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
                with open(image_path, 'wb+') as destination:
                    for chunk in image_file.chunks():
                        destination.write(chunk)
                image = format_image(image_path)
                batch.append(image.tolist())  # Convert numpy array to list for JSON serialization

            # Prepare the payload for TensorFlow Serving
            payload = {"instances": batch}
            # Make the request to TensorFlow Serving
            host = os.getenv("TENSORFLOW_SERVING_HOST")
            schema = os.getenv("SCHEMA", "http")
            headers = {"Content-Type": "application/json"}
            response = requests.post(f'{schema}://{host}/v1/models/AlzheimerDetectionBrainMRI:predict', json=payload, headers=headers)
            predictions = response.json()['predictions']
            predicted_class = tf.argmax(predictions, axis=1).numpy()    
            label_map = {0: 'Dementia', 1: 'Healthy'}
            results = []
            for index, image_file in enumerate(image_files):
                predicted_label = label_map.get(int(predicted_class[index]), "Unknown")
                results.append({
                    "image_file": image_file.name,
                    "predicted_class": predicted_label,
                    "score": predictions[index][predicted_class[index]]})

            return JsonResponse({
                "message": "Image uploaded successfully!",
                "predictions": results
            })
        else:
            return JsonResponse({"error": "Invalid file format"}, status=400)

    return render(request, "home/index.html", context)

def chat(request):
    return render(request, "home/chatbot.html")


def signup(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            # Create the user without saving to the database immediately
            new_user = form.save(commit=False)
            # Set the chosen password (hashing it)
            new_user.set_password(form.cleaned_data['password'])
            new_user.save()
            # Optionally, log the user in automatically
            return redirect('/')  # or redirect to a welcome page
    else:
        form = UserRegistrationForm()
    return render(request, 'registration/signup.html', {'form': form})

