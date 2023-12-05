# SearchPedia (https://searchpedia.pythonanywhere.com/): 

A simple search engine web app that works by indexing a set of 90k documents (based on [WikIR dataset](https://ir-datasets.com/wikir.html#wikir/en1k/training)) 
and weighted by TF-IDF and BM-25. Please note that the deployment might not work because we're using free version of Supabase and it gets freezed after some time of no usage. 
Please let me know if you want to try it out.

## How does it work?

1. Users have their information needs and therefore submit a query into SearchPedia that is handled by the Main module.
2. The Main module will contact the Indexer module to get the results to be shown to the users.
3. The Indexer module will retrieve the top-10 (hopefully) most relevant documents. We only retrieve top-10 because the latency is too much (it’s a distributed system and we use free resources available to use, can’t complain with that). Also, no re-ranking is deployed because PythonAnywhere can’t accommodate it (even after storing the docs in Supabase, it’s still not possible!)
4. As stated before, the docs contents need to be retrieved as well. The Doc Retriever module will handle that and return it to the Main module.
