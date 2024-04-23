from django.db import models

class Book(models.Model):
    release = models.DateTimeField()
    pages = models.IntegerField()
    title = models.CharField(max_length=255)
    author = models.ManyToManyField('Author')
    library = models.ForeignKey('Library', on_delete=models.CASCADE)

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




