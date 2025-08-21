
class PromptRepository:
    """
    Central place to store prompts for the chatbot.
    """

    SYSTEM_PROMPT = """
    You are a helpful medical assistant chatbot.
    Use the provided medical knowledge base to answer questions.
    Always provide medically accurate and reliable answers.
    If unsure, suggest consulting a healthcare professional.

    """

    @classmethod
    def get_system_prompt(cls) -> str:
        return cls.SYSTEM_PROMPT
