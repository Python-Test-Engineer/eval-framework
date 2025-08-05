import chromadb
from sentence_transformers import SentenceTransformer
import uuid


class SimpleRAG:
    def __init__(self, collection_name="docs", db_path="./rag_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self._get_or_create_collection(collection_name)

    def _get_or_create_collection(self, name):
        """Get existing collection or create new one"""
        collections = [col.name for col in self.client.list_collections()]
        if name in collections:
            print(f"Loading existing collection: {name}")
            return self.client.get_collection(name)
        else:
            print(f"Creating new collection: {name}")
            return self.client.create_collection(name)

    def add_documents(self, texts, metadatas=None):
        """Add documents to the collection"""
        # Check if collection already has documents
        current_count = self.collection.count()
        if current_count > 0:
            print(f"Collection already has {current_count} documents.")
            response = input("Do you want to clear and reload? (y/n): ")
            if response.lower() == "y":
                # Get all existing IDs and delete them
                all_docs = self.collection.get()
                if all_docs["ids"]:
                    self.collection.delete(ids=all_docs["ids"])
                    print("Collection cleared.")
            else:
                print("Skipping add.")
                return

        embeddings = self.embedding_model.encode(texts).tolist()
        ids = [str(uuid.uuid4()) for _ in texts]

        # Debug: Check if metadata is being passed
        print(f"Adding {len(texts)} documents with metadata: {metadatas is not None}")
        if metadatas:
            print(f"First metadata example: {metadatas[0]}")

        self.collection.add(
            documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas
        )
        print(f"Added {len(texts)} documents")

    def query(self, question):
        """Query and return 2 closest documents"""
        query_embedding = self.embedding_model.encode([question]).tolist()

        results = self.collection.query(query_embeddings=query_embedding, n_results=2)

        return {
            "question": question,
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
            "metadatas": (
                results["metadatas"][0] if results["metadatas"][0] else [None, None]
            ),
        }


# Test it
if __name__ == "__main__":
    rag = SimpleRAG()

    # Add some documents
    docs = [
        "Python is a high-level programming language created by Guido van Rossum in 1991. Known for its simple syntax and readability, Python has become one of the most popular languages for web development, data science, artificial intelligence, automation, and scientific computing. Its extensive standard library and active community make it beginner-friendly yet powerful for complex applications.",
        "Machine learning uses algorithms to automatically learn patterns from data without explicit programming instructions. This field of artificial intelligence includes supervised learning with labeled datasets, unsupervised learning for discovering hidden patterns, and reinforcement learning through trial and error. Applications span image recognition, natural language processing, recommendation systems, and predictive analytics across industries.",
        "ChromaDB is an open-source vector database specifically designed for storing and querying high-dimensional embeddings efficiently. It provides fast similarity search capabilities, supports metadata filtering, and offers both in-memory and persistent storage options. ChromaDB is particularly useful for AI applications like semantic search, recommendation engines, and retrieval-augmented generation systems.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to produce more accurate and contextually relevant AI responses. The system first searches a knowledge base for relevant documents, then uses those documents as context for generating answers. This approach significantly reduces hallucinations and provides more factual, grounded responses than pure generation models.",
        "Vector search, also known as semantic search, finds similar documents by comparing their high-dimensional embedding representations rather than relying on exact keyword matches. It uses mathematical distance metrics like cosine similarity or Euclidean distance to measure vector similarity. This enables more nuanced understanding of content meaning and context for superior search results.",
        "PostgreSQL is a powerful, open-source relational database management system that stores structured data in tables with rows and columns. It supports advanced features like JSON data types, full-text search, custom functions, and ACID compliance. PostgreSQL is widely used in web applications, data warehousing, and enterprise systems for its reliability, performance, and extensibility.",
        "SQL (Structured Query Language) is a standardized programming language designed for managing and querying relational databases. It allows users to create, read, update, and delete data using declarative statements like SELECT, INSERT, UPDATE, and DELETE. SQL is essential for database administration, data analysis, and backend development across virtually all database systems.",
        "SQLAlchemy is a popular Python library that provides a high-level Object-Relational Mapping (ORM) toolkit for working with databases. It allows developers to interact with databases using Python objects instead of raw SQL queries, while still providing access to the full power of SQL when needed. SQLAlchemy supports multiple database engines and connection pooling.",
        "Flask is a lightweight Python web framework that provides the basic tools and libraries needed to build web applications. Known for its simplicity and flexibility, Flask follows a minimalist approach, allowing developers to choose their own components and extensions. It's ideal for small to medium-sized applications, APIs, and prototyping due to its easy learning curve.",
        "FastAPI is a modern Python web framework specifically designed for building high-performance APIs quickly and efficiently. It features automatic API documentation generation, built-in data validation using Python type hints, and excellent performance comparable to Node.js and Go. FastAPI is particularly popular for building RESTful APIs and microservices with minimal boilerplate code.",
        "Streamlit is a Python library that enables data scientists and developers to create interactive web applications with minimal effort. It transforms Python scripts into shareable web apps using simple commands, without requiring HTML, CSS, or JavaScript knowledge. Streamlit is particularly popular for building data dashboards, machine learning demos, and analytical tools quickly.",
        "Dash is a Python framework for building analytical web applications, particularly suited for data visualization and interactive dashboards. Created by Plotly, it combines the power of Flask, React, and Plotly to create beautiful, interactive web apps using only Python code. Dash is widely used for business intelligence, scientific visualization, and data exploration applications.",
        "Pandas is a fundamental Python library for data manipulation and analysis, providing powerful data structures like DataFrames and Series. It offers tools for reading various file formats, cleaning messy data, performing statistical operations, and reshaping datasets. Pandas is essential for data science workflows, making complex data operations intuitive and efficient for analysts and researchers.",
    ]
    metadatas = [
        {
            "category": "programming",
            "type": "language",
            "created": 1991,
            "creator": "Guido van Rossum",
            "popularity": "high",
        },
        {
            "category": "AI",
            "type": "technique",
            "field": "machine_learning",
            "applications": "image_recognition,NLP,recommendations",
            "learning_types": "supervised,unsupervised,reinforcement",
        },
        {
            "category": "database",
            "type": "vector",
            "license": "open_source",
            "use_cases": "semantic_search,RAG,recommendations",
            "storage": "memory,persistent",
        },
        {
            "category": "AI",
            "type": "architecture",
            "field": "NLP",
            "approach": "hybrid",
            "benefits": "accuracy,factual,reduced_hallucination",
        },
        {
            "category": "search",
            "type": "technique",
            "method": "semantic",
            "metrics": "cosine_similarity,euclidean_distance",
            "advantage": "contextual_understanding",
        },
        {
            "category": "database",
            "type": "relational",
            "license": "open_source",
            "features": "JSON,full_text_search,ACID",
            "use_cases": "web_apps,data_warehouse,enterprise",
        },
        {
            "category": "database",
            "type": "language",
            "standard": "ISO_IEC_9075",
            "operations": "CREATE,READ,UPDATE,DELETE",
            "domains": "admin,analysis,backend",
        },
        {
            "category": "programming",
            "type": "library",
            "language": "Python",
            "purpose": "ORM",
            "features": "object_mapping,connection_pooling,multi_db",
        },
        {
            "category": "web",
            "type": "framework",
            "language": "Python",
            "size": "lightweight",
            "philosophy": "minimalist",
            "best_for": "small_apps,APIs,prototyping",
        },
        {
            "category": "web",
            "type": "framework",
            "language": "Python",
            "purpose": "API",
            "features": "auto_docs,validation,type_hints",
            "performance": "high",
            "architecture": "REST,microservices",
        },
        {
            "category": "web",
            "type": "library",
            "language": "Python",
            "purpose": "data_apps",
            "target_users": "data_scientists,developers",
            "use_cases": "dashboards,ML_demos,analytics",
        },
        {
            "category": "web",
            "type": "framework",
            "language": "Python",
            "creator": "Plotly",
            "purpose": "visualization",
            "technologies": "Flask,React,Plotly",
            "domains": "business_intelligence,scientific_viz,data_exploration",
        },
        {
            "category": "data",
            "type": "library",
            "language": "Python",
            "purpose": "analysis",
            "structures": "DataFrame,Series",
            "capabilities": "file_io,cleaning,statistics,reshaping",
            "users": "data_scientists,analysts,researchers",
        },
    ]
    rag.add_documents(docs, metadatas)

    # Query
    result = rag.query("What is Python?")

    print(f"\nQuestion: {result['question']}")
    print("\nClosest documents:")
    for i, (doc, distance, doc_id, metadata) in enumerate(
        zip(
            result["documents"], result["distances"], result["ids"], result["metadatas"]
        )
    ):
        print(f"{i+1}. {doc}")
        print(f"   ID: {doc_id}")
        print(f"   Distance: {distance:.3f}")
        print(f"   Metadata: {metadata}\n")
