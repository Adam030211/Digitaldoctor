from django.shortcuts import render, HttpResponse
from .models import TodoItem
from django.http import JsonResponse
from .llm_utils import get_llm_response
# views.py


def generate_text(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt', '')
        use_rag = request.POST.get('use_rag', 'true') == 'true'
        
        if prompt:
            try:
                # Call the get_llm_response function from llm_utils.py
                response = get_llm_response(prompt, use_rag=use_rag)
                return JsonResponse({'response': response})
            except Exception as e:
                return JsonResponse({'error': f'Error generating response: {str(e)}'}, status=500)
        return JsonResponse({'error': 'No prompt provided'}, status=400)
    
    # For GET requests, just render the form
    return render(request, 'generate_form.html')

# Create your views here.
def home(request):
    return render(request, "home.html")

def todos(request):
    items = TodoItem.objects.all()
    return render(request, "todos.html", {"todos": items})

