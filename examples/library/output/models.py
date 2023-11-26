from django.db import models

class Library(models.Model):
    address = models.CharField(max_length=255)
    name = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)

class Book(models.Model):
    title = models.CharField(max_length=255)
    pages = models.IntegerField()
    release = models.DateField()
    author = models.ManyToManyField('Author')
    library = models.ForeignKey('Library', on_delete=models.CASCADE)

    def __str__(self):
        return str(self.id)

class Author(models.Model):
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)
