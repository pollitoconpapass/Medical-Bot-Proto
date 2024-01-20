from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain.document_loaders import PyPDFLoader, DirectoryLoader  # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain.embeddings.cohere import CohereEmbeddings  # type: ignore
from langchain.vectorstores import FAISS  # type: ignore

  
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"


def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=25)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='distilbert-base-nli-mean-tokens', 
                                       model_kwargs= {'device': 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()
