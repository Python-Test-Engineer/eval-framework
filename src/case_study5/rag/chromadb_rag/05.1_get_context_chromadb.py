# Demo of using ChromaDB to load documents from a CSV file and query them SEMANTICALLY using metadata to refine search results so as not to search the entire document text.

# We get RAG contexts back that form part of the inputs RAGAS needs
# question, answer, contexts, grount_truth

# It might be that the contexts are the answers to the questions.

import pandas as pd
import chromadb
import json

import csv

from random import randint
from sample_data import sample_data

from rich.console import Console

console = Console()

PATH = "./chroma_db"
COLLECTION_NAME = f"reports_{randint(1000, 9999)}"
NUM_RESULTS = 1
CSV_FILE_DATA = "./src/case_study5/rag/chromadb_rag/05.2_documents.csv"
CSV_RESULTS = "./src/case_study5/rag/chromadb_rag/05.3_selected_contexts.csv"
CSV_QUESTIONS = "./src/case_study5/rag/chromadb_rag/05.0_questions.csv"

client = chromadb.PersistentClient(path=PATH)

collections = [col.name for col in client.list_collections()]

if COLLECTION_NAME in collections:
    print(f"Loading existing collection: {COLLECTION_NAME}")

else:
    print(f"Creating new collection: {COLLECTION_NAME}")


# Create a sample CSV file with 10 records
def create_sample_csv(
    file_path: str = CSV_FILE_DATA,
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
    n_results: int = NUM_RESULTS,
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
        print(f"Document: {doc[:500] + '...' if len(doc) > 200 else ''}")
        print(f"Metadata: {metadata}")
        row_result = f"{ids}|{query_text}|{doc}|{metadata}"
        with open(CSV_RESULTS, "a") as f:
            f.write(row_result + "\n")
    return results


if __name__ == "__main__":
    # Create sample CSV file
    csv_file = create_sample_csv()

    # Load data into ChromaDB
    collection = load_csv_to_chromadb(csv_file)
    list_questions = [
        # "What was the climate change about last year?",
        # "Why was food price volatility so high?",
        # "What does CRISPR-Cas9 enable?",
        # "Optimise self driving?",
    ]

    with open(CSV_QUESTIONS, "r") as f:
        for line in f:
            list_questions.append(line.strip())

    with open(CSV_RESULTS, "a") as f:
        # RAGAS uses question and contexts
        # We will add answer and ground_truth columns to complete RAGAS input requirements
        f.write("id|question|contexts|metadata\n")

    for i, query_text in enumerate(list_questions):
        console.print(f"\n[green bold]Query Text: {query_text}[/]")
        query_example(collection, query_text=query_text)

    print("\nScript completed successfully!")
