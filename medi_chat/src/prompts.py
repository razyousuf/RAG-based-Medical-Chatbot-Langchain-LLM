"""
Module: prompts.py
Description: Provides structured and reusable prompt templates for the RAG-based medical assistant.
"""

from typing import Literal


class PromptTemplate:
    """
    A class to manage system prompts for different components of the assistant.
    Currently supports the default medical QA assistant prompt.
    """

    def __init__(self, prompt_type: Literal["medical_assistant"] = "medical_assistant"):
        self.prompt_type = prompt_type
        self.templates = {
            "medical_assistant": (
                "You are a medical assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise.\n\n{context}"
            )
        }

    def get_prompt(self) -> str:
        """Returns the selected prompt template."""
        try:
            return self.templates[self.prompt_type]
        except KeyError:
            raise ValueError(f"Unsupported prompt type: {self.prompt_type}")

    def format_prompt(self, context: str) -> str:
        """
        Injects dynamic context into the prompt template.

        Args:
            context (str): The context to embed in the prompt.

        Returns:
            str: The formatted prompt ready for the LLM.
        """
        return self.get_prompt().format(context=context)
