from django.db import models

class Author(models.Model):
    email = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    book = models.ManyToManyField('Book')

    def __str__(self):
        return str(self.id)

class Book(models.Model):
    pages = models.IntegerField()
    release = models.DateTimeField()
    title = models.CharField(max_length=255)
    library = models.ForeignKey('Library', on_delete=models.CASCADE)

    def __str__(self):
        return str(self.id)

class Library(models.Model):
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)




