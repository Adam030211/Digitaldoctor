# management/commands/process_documents.py
from django.core.management.base import BaseCommand
from myapp.embeddings import create_embeddings_for_chunks
from myapp.models import Document

class Command(BaseCommand):
    help = 'Process documents and create embeddings'

    def handle(self, *args, **options):
        # Count documents without content
        docs_without_content = Document.objects.filter(content__isnull=True).count()
        if docs_without_content:
            self.stdout.write(f"Found {docs_without_content} documents without content")
        
        # Create embeddings
        chunks_processed = create_embeddings_for_chunks()
        self.stdout.write(self.style.SUCCESS(f"Created embeddings for {chunks_processed} chunks"))