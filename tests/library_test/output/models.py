from django.db import models

class Book(models.Model):
    release = models.DateTimeField()
    title = models.CharField(max_length=255)
    pages = models.IntegerField()
    library = models.ForeignKey('Library', on_delete=models.CASCADE)
    author = models.ManyToManyField('Author')

    def __str__(self):
        return str(self.id)

class Library(models.Model):
    address = models.CharField(max_length=255)
    name = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)

class Author(models.Model):
    email = models.CharField(max_length=255)
    name = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)




