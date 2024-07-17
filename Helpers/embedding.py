from langchain_community.embeddings.ollama import OllamaEmbeddings


def embedding_function():
    embedding = OllamaEmbeddings(model='mistral')
    return embedding
