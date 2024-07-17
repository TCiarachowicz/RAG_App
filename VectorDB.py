from langchain.vectorstores import Chroma
from Helpers.embedding import embedding_function
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


class VectorDB:
    def __init__(self, upload_folder):
        self.chroma_path = 'vectordb'
        self.data_path = upload_folder
        self.db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=embedding_function()
        )

    def load_documents(self):
        document_loader = PyPDFDirectoryLoader(self.data_path)
        return document_loader.load()

    @staticmethod
    def split_documents(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def add_new_source(self):
        documents = self.load_documents()
        chunks = self.split_documents(documents)

        last_page_id = None
        current_chunk_index = 0
        chunks_ids = []

        for chunk in chunks:
            source = chunk.metadata.get('source')
            page = chunk.metadata.get('page')
            current_page_id = f'{source}:{page}'

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk.metadata['id'] = f'{current_page_id}:{current_chunk_index}'
            last_page_id = current_page_id
            chunks_ids.append(chunk.metadata['id'])

        self.db.add_documents(chunks, ids=chunks_ids)

    def nb_of_existing_doc(self):
        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items['ids'])
        return len(existing_ids)
