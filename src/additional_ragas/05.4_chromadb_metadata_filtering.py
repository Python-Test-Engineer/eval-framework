import chromadb

PATH = "./chroma_db"
COLLECTION_NAME = "reports"


client = chromadb.PersistentClient(path=PATH)  # or HttpClient()
collections = client.list_collections()
print("=" * 70)
print(f"Existing collections: {collections}")
print("=" * 70)
# Initialize ChromaDB client with the latest configuration
client = chromadb.PersistentClient(path=PATH)

# Create or get a collection
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Example: Adding documents with metadata
collection.add(
    documents=[
        "Document about web development",
        "Document about telecom services",
        "Another document about Alltel services",
    ],
    metadatas=[
        {"year": 2022, "category": "web", "domain": "alltel"},
        {"year": 2023, "category": "telecom", "domain": "alltel"},
        {"year": 2022, "category": "telecom", "domain": "alltel"},
    ],
    ids=["doc1", "doc2", "doc3"],
)

# Example 1: Filter by exact metadata match (all fields)
# Using $and to combine multiple exact matches
results = collection.query(
    query_texts=["services"],
    where={
        "$and": [
            {"year": {"$eq": 2022}},
            {"category": {"$eq": "web"}},
            {"domain": {"$eq": "alltel"}},
        ]
    },
    n_results=10,
)
print("Example 1 - Filter by exact metadata match:")
print(results)

# Example 2: Filter by one metadata field
results = collection.query(
    query_texts=["services"], where={"domain": {"$eq": "alltel"}}, n_results=10
)
print("\nExample 2 - Filter by domain:")
print(results)

# Example 3: Filter with comparison operators
results = collection.query(
    query_texts=["services"], where={"year": {"$eq": 2022}}, n_results=10
)
print("\nExample 3 - Filter by year equal to 2022:")
print(results)

# Example 4: Filter with multiple conditions using $and
results = collection.query(
    query_texts=["services"],
    where={"$and": [{"year": {"$eq": 2022}}, {"domain": {"$eq": "alltel"}}]},
    n_results=10,
)
print("\nExample 4 - Filter with $and operator:")
print(results)

# Example 5: Filter with $or operator
results = collection.query(
    query_texts=["services"],
    where={"$or": [{"category": {"$eq": "web"}}, {"category": {"$eq": "telecom"}}]},
    n_results=10,
)
print("\nExample 5 - Filter with $or operator:")
print(results)

# Example 6: More complex filtering with nested conditions
results = collection.query(
    query_texts=["services"],
    where={
        "$and": [
            {"domain": {"$eq": "alltel"}},
            {"$or": [{"year": {"$eq": 2022}}, {"category": {"$eq": "telecom"}}]},
        ]
    },
    n_results=10,
)
print("\nExample 6 - Complex nested filtering:")
print(results)

# Example 7: Using $in operator
results = collection.query(
    query_texts=["services"], where={"year": {"$in": [2021, 2022]}}, n_results=10
)
print("\nExample 7 - Filter using $in operator:")
print(results)

# Example 8: Using $nin (not in) operator
results = collection.query(
    query_texts=["services"], where={"year": {"$nin": [2023, 2024]}}, n_results=10
)
print("\nExample 8 - Filter using $nin operator:")
print(results)

# Example 9: Using numeric comparisons
results = collection.query(
    query_texts=["services"],
    where={"$and": [{"year": {"$gt": 2021}}, {"year": {"$lt": 2023}}]},
    n_results=10,
)
print("\nExample 9 - Filter using numeric comparisons:")
print(results)

# Example 10: Get all documents with specified metadata
results = collection.get(
    where={
        "$and": [
            {"year": {"$eq": 2022}},
            {"category": {"$eq": "web"}},
            {"domain": {"$eq": "alltel"}},
        ]
    }
)
print("\nExample 10 - Get all documents with exact metadata match:")
print(results)
