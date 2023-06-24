from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from config import embedding_models

MODEL_NAME = "all-MiniLM-L6-v2"


def get_retriever(dataset):
    embedding = HuggingFaceEmbeddings(model_name=embedding_models[MODEL_NAME]['name'],
                                      model_kwargs=embedding_models[MODEL_NAME]['kwargs'])
    vector_db = Chroma.from_documents(documents=dataset,
                                      embedding=embedding)
    return vector_db.as_retriever(search_kwargs={"k": 3})


