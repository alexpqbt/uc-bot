from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from config import vector_store 
from pathlib import Path
import time
import json

json_file = Path("./data/cleaned_pages.json")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

def send_batches(splits, batch_size = 25, delay = 61):
    total_batches = (len(splits) + batch_size - 1) // batch_size

    for i in range(0, len(splits), batch_size):
        batch_num = (i // batch_size) + 1

        batch = splits[i:i + batch_size]

        start_time = time.time()

        vector_store.add_documents(documents=batch)

        elapsed = time.time() - start_time

        print(
            f"Processed batch {batch_num}/{total_batches} "
            f"({len(batch)} docs) in {elapsed:.2f} seconds"
        )

        if i + batch_size < len(splits):
            for j in range(delay):
                print(f"Honk shoo{'.' * j}", end="\r", flush=True)
                time.sleep(1)
            print()  # move to a new line when done

def metadata_func(record, metadata):
    metadata["title"] = record.get("title")
    metadata["url"] = record.get("url")

    return metadata

with open(json_file, "r") as f:
    data = json.load(f)

loader = JSONLoader(
    file_path=json_file,
    jq_schema=".[]",
    content_key="content",
    metadata_func=metadata_func
)

docs = loader.load()
splits = text_splitter.split_documents(docs)

send_batches(splits)