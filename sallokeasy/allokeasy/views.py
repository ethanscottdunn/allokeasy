from django.shortcuts import get_object_or_404, render

# Create your views here.
# polls/views.py

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import csv
import io
from django.http import JsonResponse
from utils.vanguard import parse_cost_basis_vanguard_csv_by_ticker
import json

def index(request):
    #return HttpResponse("Hello, world. You're at the allokeasy index.")
    return render(request,"allokeasy/index.html")

@csrf_exempt
def upload_csv(request):
    if request.method == "POST" and request.FILES.get("file"):
        csv_file = request.FILES["file"]
        print("got into the function")
       
        # Read file in memory
        decoded_file = csv_file.read().decode("utf-8")
        io_string = io.StringIO(decoded_file)
        #reader = csv.DictReader(io_string)

        #alloc_dict = parse_cost_basis_vanguard_csv_by_ticker(io_string)
        alloc_dict_list = parse_cost_basis_vanguard_csv_by_ticker(io_string)
        # ticker, quantity, market_value

        #allocations = []
        #for k, v in alloc_dict.items():
        #    # Expect columns: ticker, value
        #    allocations.append({
        #        "ticker": k,
        #        "value": f"{v:.2f}"
        #    })
        #print(allocations)
        #print((type(allocations)))

        #return JsonResponse({"allocations": allocations})
        return JsonResponse({"allocations": alloc_dict_list})

    return JsonResponse({"error": "No file uploaded"}, status=400)
