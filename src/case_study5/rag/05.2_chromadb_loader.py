# Demo of using ChromaDB to load documents from a CSV file and query them SEMANTICALLY using metadata to refine search results so as not to search the entire document text.

# It will become more useful later on when we do some Data Analytics but can be used for RAG as in FAQ, even taking in a filter returned from ROUTER.

# ROUTER can provide the metadata to filter the results, e.g. by source, author, year, etc. and then we can use the metadata to refine the search results in ChromaDB to be the In Context Learning (ICL) for the RAG of FAQ.

import pandas as pd
import chromadb
import json
import uuid
import csv
from typing import List, Dict, Any
from random import randint
from sample_data import sample_data

PATH = "./chroma_db"
COLLECTION_NAME = f"reports_{randint(1000, 9999)}"

client = chromadb.PersistentClient(path=PATH)

collections = [col.name for col in client.list_collections()]

if COLLECTION_NAME in collections:
    print(f"Loading existing collection: {COLLECTION_NAME}")

else:
    print(f"Creating new collection: {COLLECTION_NAME}")


# Create a sample CSV file with 10 records
def create_sample_csv(
    file_path: str = "./src/case_study5/rag/05.3_chromadb_loader.csv",
):
    """
    Create a sample CSV file with 10 records containing id, document, and metadata columns.
    The metadata column contains a JSON string representing metadata used by ChromaDB.
    """

    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "document", "metadata"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_data)

    print(f"Sample CSV created at {file_path}")
    return file_path


def load_csv_to_chromadb(csv_file: str, collection_name: str = COLLECTION_NAME):
    """
    Load documents from a CSV file into ChromaDB.

    The CSV should have columns:
    - id: UUID for document identification
    - document: The text content
    - metadata: JSON string containing metadata

    Note: ChromaDB metadata values must be strings, integers, floats, or booleans.
    Lists and dictionaries are not supported as direct values in metadata.
    """
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=PATH)

    # Create or get a collection
    try:
        collection = client.create_collection(name=COLLECTION_NAME)
        print(f"Created new collection: {COLLECTION_NAME}")
    except ValueError:
        # Collection already exists
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []

    for _, row in df.iterrows():
        # Get UUID from CSV
        doc_id = row["id"]

        # Get document text
        document = row["document"]

        # Parse metadata from JSON string
        try:
            metadata_raw = json.loads(row["metadata"])

            # Ensure all metadata values are valid for ChromaDB (strings, ints, floats, booleans)
            # Convert any list/dict values to strings if necessary
            metadata = {}
            for key, value in metadata_raw.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif isinstance(value, (list, dict)):
                    # Convert lists/dicts to strings to comply with ChromaDB requirements
                    metadata[key] = json.dumps(value)
                else:
                    # Skip None or unsupported types
                    metadata[key] = str(value)

        except json.JSONDecodeError:
            print(
                f"Warning: Invalid JSON metadata for document {doc_id}. Using empty metadata."
            )
            metadata = {}

        ids.append(doc_id)
        documents.append(document)
        metadatas.append(metadata)

    # Add documents to collection in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end], documents=documents[i:end], metadatas=metadatas[i:end]
        )
        print(f"Added batch {i//batch_size + 1}: documents {i+1} to {end}")

    # Print collection stats
    print(f"Total documents in collection: {len(collection.get(include=[])['ids'])}")
    return collection


def query_example(
    collection,
    query_text: str = "How did renewable investments do last year?",
    n_results: int = 3,
):
    """Run a sample query against the loaded collection"""
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nQuery Results for '{query_text}':")
    for i, (ids, doc, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        print(f"\nResult No: {i+1} \nid: {ids} \ndistance: {distance:.4f}")
        print(f"Document: {doc[:100] + '...' if len(doc) > 100 else ''}")
        print(f"Metadata: {metadata}")

    return results


if __name__ == "__main__":
    # Create sample CSV file
    csv_file = create_sample_csv()

    # Load data into ChromaDB
    collection = load_csv_to_chromadb(csv_file)
    query_example(collection, query_text="What was the climate change about last year?")
    # Run an example query
    # query_example(collection)

    print("\nScript completed successfully!")
