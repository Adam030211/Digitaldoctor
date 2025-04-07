from django.db import models

# Create your models here.

class TodoItem(models.Model):
    title = models.CharField(max_length=200)
    completed = models.BooleanField(default=False)


class Document(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    document_type = models.CharField(max_length=50, default="PDF")
    content = models.TextField(blank=True, null=True)  # To store extracted text
    date_added = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title

class DocumentChunk(models.Model):
    """Smaller chunks of documents for better retrieval"""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField()
    chunk_index = models.IntegerField()
    embedding = models.JSONField(null=True, blank=True)  # Store vector as JSON
    
    class Meta:
        ordering = ['document', 'chunk_index']