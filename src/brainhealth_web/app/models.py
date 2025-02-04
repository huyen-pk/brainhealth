from django.db import models

class UploadedFiles(models.Model):
    files = models.FileField(upload_to="uploads/")  # Saves files to media/uploads/
    uploaded_at = models.DateTimeField(auto_now_add=True)
    patient_name = models.CharField(max_length=255, name="patient_name")
    patient_id = models.CharField(max_length=255, name="patient_id")

    def __str__(self):
        return f"File {self.id}"
