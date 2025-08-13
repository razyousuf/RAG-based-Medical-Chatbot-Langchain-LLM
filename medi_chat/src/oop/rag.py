
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from medi_chat.src.oop.exception import AppException
from medi_chat.src.oop.logger import logger
import sys

class RAGPipeline:
    """RAG pipeline wrapping retriever, LLM and prompt into a callable interface."""

    def __init__(self, vectorstore, openai_model: str, system_prompt: str, k: int = 3):
        try:
            self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
            self.llm = ChatOpenAI(model=openai_model)
            self.prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
            self.qa_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, self.qa_chain)
            logger.debug("Initialized RAGPipeline")
        except Exception as e:
            logger.error("Failed to initialize RAGPipeline", exc_info=True)
            raise AppException(e, sys)

    def answer(self, question: str) -> str:
        try:
            resp: Dict[str, Any] = self.rag_chain.invoke({"input": question})
            answer = resp.get("answer", "")
            logger.info("Generated answer successfully")
            return answer
        except Exception as e:
            logger.error("Failed during RAG answer", exc_info=True)
            raise AppException(e, sys)
