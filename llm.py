from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings

from doc_loader import constants

load_dotenv()

from langchain_openai import ChatOpenAI

from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.chroma import Chroma


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=constants.CHROMA_DB,
        embedding_function=embeddings,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=db.as_retriever(), return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":

    generated = run_llm(
        query="What is the condition associated with Shoulder Pain and Mobility Deficits?"
    )
    print(generated)
