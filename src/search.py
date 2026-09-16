import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:
    def __init__(self,persist_dir: str = "faiss_store",embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",llm_model: str = "openai/gpt-oss-20b",data_dir: str = "data",):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print(f"[INFO] No existing index found at '{persist_dir}', building from '{data_dir}'...")
            docs = load_all_documents(data_dir)
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example) or export it as an environment variable."
            )

        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def _retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        results = self.vectorstore.query(query, top_k=top_k)
        sources = []
        for r in results:
            meta = r.get("metadata") or {}
            text = meta.get("text", "")
            if text:
                sources.append({"text": text, "distance": float(r["distance"])})
        return sources

    def search_with_sources(self, query: str, top_k: int = 5) -> Tuple[str, List[dict]]:
        """Retrieve relevant chunks and generate a grounded answer.

        Returns (answer, sources) so the caller (e.g. the Streamlit UI)
        can display citations alongside the generated answer.
        """
        sources = self._retrieve(query, top_k=top_k)
        if not sources:
            return "No relevant documents found.", []

        context = "\n\n".join(s["text"] for s in sources)
        prompt = (
            "Answer the following question using only the provided context. "
            "If the context does not contain the answer, say so explicitly.\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        )
        response = self.llm.invoke([prompt])
        return response.content, sources

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Backwards-compatible wrapper around search_with_sources."""
        answer, _ = self.search_with_sources(query, top_k=top_k)
        return answer


# Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     answer, sources = rag_search.search_with_sources("What is BERT?", top_k=3)
#     print("Answer:", answer)
#     print(f"Backed by {len(sources)} source chunks.")
