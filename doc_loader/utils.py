from langchain.schema import Document
import json
from typing import Iterable, List


def save_docs_to_jsonl(documents: List[Document], file_path: str) -> None:
    with open(file_path, "w") as jsonl_file:
        for doc in documents:
            jsonl_file.write(doc.json() + "\n")


def load_docs_from_jsonl(file_path: str) -> List[Document]:
    documents = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            documents.append(obj)
    return documents
