#views.py
from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from .llm_utils import get_llm_response
import traceback

def generate_text(request):
    if request.method == 'POST':
        prompt = request.POST.get('prompt', '')
        use_rag = request.POST.get('use_rag', 'true') == 'true'
        
        if prompt:
            try:
                response = get_llm_response(request, prompt, use_rag=use_rag)
                return JsonResponse({'response': response})
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
        return JsonResponse({'error': 'No prompt provided'}, status=400)
    return render(request, 'generate_form.html')


# Create your views here.
def test(request):

    if request.method == 'POST':
        prompt = request.POST.get('prompt', '')
        use_rag = request.POST.get('use_rag', 'true') == 'true'
        
        if prompt:
            try:
                response = get_llm_response(request, prompt, use_rag=use_rag)
                return JsonResponse({'response': response})
            except Exception as e:
                print(traceback.format_exc())
                return JsonResponse({'error': str(e)}, status=500)
        return JsonResponse({'error': 'No prompt provided'}, status=400)
    return render(request, "home.html")



