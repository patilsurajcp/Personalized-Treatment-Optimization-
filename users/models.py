from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
import os
# Create your models here.

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='user.png', upload_to='profile_pics')

    def __str__(self):
        return f'{self.user.username} Profile'
    