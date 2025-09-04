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
        # TODO: put the file back in the import area...
        return JsonResponse({"allocations": alloc_dict_list})

    return JsonResponse({"error": "No file uploaded"}, status=400)

@csrf_exempt
def produce_optimized_portfolio(request):
    if request.method == "POST" and request.FILES.get("file"):
        csv_file = request.FILES["file"]
        # Read file in memory
        decoded_file = csv_file.read().decode("utf-8")
        io_string = io.StringIO(decoded_file)
        ## TODO: push this data into the function that computes new allocations and stats
        alloc_dict_list = parse_cost_basis_vanguard_csv_by_ticker(io_string)

    # Handle other input values
    try:
        income = float(request.POST.get("income"))
        risk_free_rate = float(request.POST.get("risk_free_rate"))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid inputs for income and/or risk free rate."}, status=400)

    # Handle dropdown
    filing_status = request.POST.get("filing_status")
    if filing_status not in ["single", "married_joint", "married_separate"]:
        return JsonResponse({"error": "Invalid filing status."}, status=400)

    # start_date & end_date
    try:
        start_date = request.POST.get("start_date")
        end_date = request.POST.get("end_date")
    except:
        return JsonResponse({"error": "Invalid inputs for start and/or end date"})

    ## TODO: run the optimization.
    ## For now, just print the values so I know they came in right.
    #print(alloc_dict_list)
    print("income: %s" % income)
    print("filing status: %s" % filing_status)
    print("risk_free_rate: %s" % risk_free_rate)
    print("start_date: %s" % start_date)
    print("end_date: %s" % end_date)

    #if request.method == "POST" and request.FILES.get("file"):
        #return JsonResponse({"allocations": alloc_dict_list})
    return JsonResponse({"error": "information received"}, status=400)



