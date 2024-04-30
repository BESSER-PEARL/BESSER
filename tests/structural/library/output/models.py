from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    publishes = models.ManyToManyField('Book', blank=True, null=True)

    def __str__(self):
        return str(self.id)

class Library(models.Model):
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)

class Book(models.Model):
    title = models.CharField(max_length=255)
    pages = models.IntegerField()
    release = models.DateTimeField()
    locatedIn = models.ForeignKey('Library')

    def __str__(self):
        return str(self.id)




