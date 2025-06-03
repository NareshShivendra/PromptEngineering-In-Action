import os
import nltk
import numpy as np
from dotenv import load_dotenv
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory


# Load environment variables (for OpenAI API Key)
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai_api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------
# Step 1: Load Reuters dataset
# -------------------------------
def load_reuters_docs(limit=500):
    nltk.download('reuters')
    from nltk.corpus import reuters
    file_ids = reuters.fileids()
    docs_raw = [reuters.raw(fid) for fid in file_ids[:limit]]
    documents = [Document(page_content=doc, metadata={"source": fid}) for doc, fid in zip(docs_raw, file_ids[:limit])]
    print(f"Loaded {len(documents)} Reuters documents.")
    return documents

# -------------------------------
# Step 2: Create vector store with FAISS
# -------------------------------
def create_vector_store(documents):
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store, embedding_model

# -------------------------------
# Step 3: Ask questions and use OpenAI directly
# -------------------------------
def run_queries(embedding_model, documents, queries):
    long_term_memory = []

    for query in queries:
        # Enhance query with long-term memory context
        enhanced_query = query
        if long_term_memory:
            query_embedding = embedding_model.embed_query(query)
            memory_texts = [f"User: {m['query']}\nAssistant: {m['response']}" for m in long_term_memory]
            memory_embeddings = [embedding_model.embed_query(text) for text in memory_texts]
            similarities = [np.dot(query_embedding, mem_emb) for mem_emb in memory_embeddings]
            memory_context = ""
            for idx in np.argsort(similarities)[-2:]:
                if similarities[idx] > 0.75:
                    memory_context += f"Previously discussed:\nQ: {long_term_memory[idx]['query']}\nA: {long_term_memory[idx]['response']}\n\n"
            if memory_context:
                enhanced_query = memory_context + f"\nCurrent question: {query}"

        # Embed and retrieve top-k similar documents
        query_embedding = embedding_model.embed_query(enhanced_query)
        results = FAISS.from_documents(documents, embedding_model)
        retriever = results.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(enhanced_query)
        sources = [doc.metadata['source'] for doc in retrieved_docs]
        context = "\n\n".join([f"[{doc.metadata['source']}]: {doc.page_content[:300]}..." for doc in retrieved_docs])

        # Call OpenAI API with context + question
        prompt = f"""
        You are a financial analyst. Given the query and relevant document snippets below, write a clear and concise answer.

        Query:
        {enhanced_query}

        Context:
        {context}

        Answer:
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial assistant that provides concise, insightful answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"OpenAI API Error: {e}"

        # Output result
        print(f"\nUser: {query}")
        print(f"AI: {answer}")
        print(f"Sources: {sources}")

        # Store in long-term memory if long enough
        if len(answer) > 300:
            if long_term_memory:
                similarities = [
                    np.dot(query_embedding, m["embedding"]) for m in long_term_memory
                ]
                if max(similarities) < 0.85:
                    long_term_memory.append({
                        "query": query,
                        "response": answer,
                        "embedding": query_embedding
                    })
            else:
                long_term_memory.append({
                    "query": query,
                    "response": answer,
                    "embedding": query_embedding
                })
        if len(long_term_memory) > 50:
            long_term_memory = long_term_memory[-50:]

# -------------------------------
# Execution of MemoRAG
# -------------------------------
def main():
    documents = load_reuters_docs(limit=500)
    vector_store, embedding_model = create_vector_store(documents)
    queries = [
        "What is driving oil prices in the last decade?",
        "Analyze how oil price fluctuations have influenced stock markets, currency valuations, and international trade over the past decade. Highlight the causal relationships among these factors. Can you also highlight the points to support the response?"
    ]
    run_queries(embedding_model, documents, queries)

if __name__ == "__main__":
    main()  
