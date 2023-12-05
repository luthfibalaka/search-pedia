from django.shortcuts import render
from indexing.bsbi import BSBIIndex
from indexing.compression import VBEPostings
from supabase import create_client, Client

import os

BSBI_instance = BSBIIndex(
    data_dir="collections", postings_encoding=VBEPostings, output_dir="index"
)

url_01: str = "secret"
key_01: str = "secret"
url_45: str = "secret"
key_45: str = "secret"
url_23: str = "secret"
key_23: str = "secret"


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
