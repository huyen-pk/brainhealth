from django import forms
from .models import UploadedFiles
from django.contrib.auth.models import User

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedFiles
        fields = ["files", "patient_name", "patient_id"]


class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput)
    password_confirm = forms.CharField(label='Confirm Password', widget=forms.PasswordInput)
    
    class Meta:
        model = User
        fields = ('username', 'email')
    
    def clean_password_confirm(self):
        cd = self.cleaned_data
        if cd.get('password') != cd.get('password_confirm'):
            raise forms.ValidationError('Passwords don’t match.')
        return cd.get('password_confirm')
