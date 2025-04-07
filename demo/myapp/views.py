from django.shortcuts import render, HttpResponse
from .models import TodoItem
from django.http import JsonResponse
from .llm_utils import get_llm_response
# views.py
from .models import Document
from .scraper import scrape_documents

def generate_text(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt', '')
        use_rag = request.POST.get('use_rag', 'true') == 'true'
        
        if prompt:
            response = get_llm_response(prompt, use_rag=use_rag)
            return JsonResponse({'response': response})
        return JsonResponse({'error': 'No prompt provided'}, status=400)
    
    # Get document count for the template
    doc_count = Document.objects.count()
    return render(request, 'generate_form.html', {'doc_count': doc_count})

# Create your views here.
def home(request):
    return render(request, "home.html")

def todos(request):
    items = TodoItem.objects.all()
    return render(request, "todos.html", {"todos": items})

