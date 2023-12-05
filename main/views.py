from django.shortcuts import render
from indexing.bsbi import BSBIIndex
from indexing.compression import VBEPostings
from supabase import create_client, Client

import os

BSBI_instance = BSBIIndex(
    data_dir="collections", postings_encoding=VBEPostings, output_dir="index"
)

url_01: str = "https://jzlmjbjpfpblgvznyayl.supabase.co"
key_01: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp6bG1qYmpwZnBibGd2em55YXlsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwMTYwMTM2MywiZXhwIjoyMDE3MTc3MzYzfQ.8Gnr-0gJV1fCFxvwj7BmYGwP1vD4Ihwi_si8RsNO2f0"

url_45: str = "https://nwyrkbmosxlcgclvrljr.supabase.co"
key_45: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im53eXJrYm1vc3hsY2djbHZybGpyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwMTY1OTcwMCwiZXhwIjoyMDE3MjM1NzAwfQ.UDVqH2bhk9uVqBkiWiR4a1jSxgCkLWbzCbbylr93n5c"

url_23: str = "https://wuhxqndlflxejzndltia.supabase.co"
key_23: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind1aHhxbmRsZmx4ZWp6bmRsdGlhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcwMTY1ODgxMywiZXhwIjoyMDE3MjM0ODEzfQ.jItqpYd0hgVj9uSuTSU6T1ydD4ahBjCbdLeXWYPvjBI"


def index(request):
    return render(request, "index.html")


def serp(request):
    query = request.GET["query"]
    search_results = BSBI_instance.retrieve_bm25(query, k=10)
    for i in range(len(search_results)):
        result = search_results[i]
        node = result[1].split("/")[1]
        doc_id = result[1].split("/")[2].split(".")[0]

        if node == "0" or node == "1":
            supabase = create_client(url_01, key_01)
            content = supabase.table('documents').select(f"text").eq("doc_id", doc_id).execute()
        elif node == "2" or node == "3":
            supabase = create_client(url_23, key_23)
            content = supabase.table('documents').select(f"text").eq("doc_id", doc_id).execute()
        else:
            supabase = create_client(url_45, key_45)
            content = supabase.table('documents').select(f"text").eq("doc_id", doc_id).execute()
        search_results[i] = (result[0], result[1], content.data[0]['text'])
    context = {}
    context["query"] = query
    context["results"] = search_results
    return render(request, "serp.html", context)
