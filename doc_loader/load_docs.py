import glob
import os
from typing import List, Union, Type

from dotenv import load_dotenv
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
    UnstructuredPDFLoader,
)
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings,
    OpenAIEmbeddings,
)
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

from doc_loader import constants
from doc_loader.utils import save_docs_to_jsonl, load_docs_from_jsonl

load_dotenv()

doc_types = ("docx", "doc", "pdf")

EMBEDDING_TYPE = Union[Type[SentenceTransformerEmbeddings], Type[OpenAIEmbeddings]]

SPLITTER_TYPE = Union[Type[CharacterTextSplitter], Type[RecursiveCharacterTextSplitter]]


def load_docs_from_dir(root_dir: str, show_progress: bool = False):
    documents: List[Document] = []

    if "pdf" in doc_types:
        print("Start to load pdf files...")
        pdf_loader = DirectoryLoader(
            root_dir,
            glob="**/*.pdf",
            loader_cls=UnstructuredPDFLoader,
            show_progress=show_progress,
        )
        pdf_documents = pdf_loader.load()
        documents.extend(pdf_documents)

    if "docx" in doc_types:
        print("Start to load docx files...")
        docx_loader = DirectoryLoader(
            root_dir,
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=show_progress,
        )
        docx_documents = docx_loader.load()
        documents.extend(docx_documents)

    if "doc" in doc_types:
        print("Start to load doc files...")
        doc_loader = DirectoryLoader(
            root_dir,
            glob="**/*.doc",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=show_progress,
        )
        doc_documents = doc_loader.load()
        documents.extend(doc_documents)

    # TODO: add loaders for other types

    return documents


def split_docs(
    documents: List[Document],
    splitter_cls: SPLITTER_TYPE,
    chunk_size: int,
    chunk_overlap: int,
    **splitter_kwargs,
):
    splitter = splitter_cls(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, **splitter_kwargs
    )
    split_docs: List[Document] = splitter.split_documents(documents)
    return split_docs


def create_embeddings(
    documents: List[Document],
    embedding_cls: EMBEDDING_TYPE,
    model_name: str,
):
    embedding_function = embedding_cls(model_name=model_name)
    Chroma.from_documents(
        documents, embedding_function, persist_directory=constants.CHROMA_DB
    )


def test_embeddings(
    query: str,
    embedding_cls: EMBEDDING_TYPE,
    model_name: str,
    persist_directory: str,
):
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_cls(model_name=model_name),
    )
    docs = db.similarity_search(query)
    print(docs)


def pipeline(mode: str):
    if mode == "end_to_end":
        documents = load_docs_from_dir(root_dir=constants.ROOT_DIR)
        split_documents = split_docs(
            documents=documents,
            splitter_cls=RecursiveCharacterTextSplitter,
            chunk_size=1000,
            chunk_overlap=100,
        )
        create_embeddings(
            documents=split_documents,
            embedding_cls=SentenceTransformerEmbeddings,
            model_name="all-MiniLM-L6-v2",
        )


if __name__ == "__main__":
    root_dir = constants.ROOT_DIR
    # # need to install libreoffice to system
    documents = load_docs_from_dir(root_dir, show_progress=True)
    for i, document in enumerate(documents):
        print(f"document {i}: {document.metadata}")
    save_docs_to_jsonl(documents=documents, file_path=constants.DOCUMENTS_PATH)
    documents = load_docs_from_jsonl(file_path=constants.DOCUMENTS_PATH)
    split_documents = split_docs(
        documents=documents,
        splitter_cls=RecursiveCharacterTextSplitter,
        chunk_size=600,
        chunk_overlap=100,
    )
    print(len(split_documents))
    save_docs_to_jsonl(
        documents=split_documents, file_path=constants.SPLIT_DOCUMENTS_PATH
    )
    #
    split_documents = load_docs_from_jsonl(file_path=constants.SPLIT_DOCUMENTS_PATH)
    print("Start to create embeddings...")
    create_embeddings(
        documents=split_documents,
        embedding_cls=SentenceTransformerEmbeddings,
        model_name="all-MiniLM-L6-v2",
    )
    print("Finished creating embeddings.")
    test_embeddings(
        query="What is the condition associated with Shoulder Pain and Mobility Deficits?",
        persist_directory=constants.CHROMA_DB,
        embedding_cls=SentenceTransformerEmbeddings,
        model_name="all-MiniLM-L6-v2",
    )
