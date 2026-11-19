"""Load the persisted FAISS policy store as a LangChain retriever."""
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL, VECTOR_STORE_DIR


def get_store() -> FAISS:
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        import ingest
        ingest.main()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
    return FAISS.load_local(
        str(VECTOR_STORE_DIR), embeddings,
        allow_dangerous_deserialization=True,  # we created this index ourselves
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def get_retriever(k: int = 5, store: FAISS | None = None):
    store = store or get_store()
    return store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 15})
