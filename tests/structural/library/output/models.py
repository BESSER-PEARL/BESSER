from django.db import models

class Library(models.Model):
    address = models.CharField(max_length=255)
    name = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)

class Book(models.Model):
    pages = models.IntegerField()
    release = models.DateTimeField()
    title = models.CharField(max_length=255)
    author = models.ManyToManyField('Author')
    library = models.ForeignKey('Library', on_delete=models.CASCADE)

    def __str__(self):
        return str(self.id)

class Author(models.Model):
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)




