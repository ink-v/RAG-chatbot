def get_context_prompt(language: str) -> str:
    if language == "span":
        return CONTEXT_PROMPT_ES
    return CONTEXT_PROMPT_EN


def get_condensed_context_prompt(language: str) -> str:
    if language == "span":
        return CONDENSED_CONTEXT_PROMPT_ES
    return CONDENSED_CONTEXT_PROMPT_EN


def get_system_prompt(language: str, is_rag_prompt: bool = True) -> str:
    if language == "span":
        return SYSTEM_PROMPT_RAG_ES if is_rag_prompt else SYSTEM_PROMPT_ES
    return SYSTEM_PROMPT_RAG_EN if is_rag_prompt else SYSTEM_PROMPT_EN


SYSTEM_PROMPT_EN = """\
This is a chat between a user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. \
The assistant should also indicate when the answer cannot be found in the context."""

SYSTEM_PROMPT_RAG_EN = """\
This is a chat between a user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. \
The assistant should also indicate when the answer cannot be found in the context."""

CONTEXT_PROMPT_EN = """\
Here are the relevant documents for the context:

{context_str}

Instruction: Based on the above documents, provide a detailed answer for the user question below. \
Answer 'don't know' if not present in the document."""

CONDENSED_CONTEXT_PROMPT_EN = """\
Given the following conversation between a user and an AI assistant and a follow up question from user,
rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:\
"""

SYSTEM_PROMPT_ES = """\
Esta es una conversación entre un usuario y un asistente de inteligencia artificial. \
El asistente proporciona respuestas útiles, detalladas y corteses a las preguntas del usuario basadas en el contexto. \
El asistente también debe indicar cuando la respuesta no se puede encontrar en el contexto. \
Regla obligatoria: responde siempre y únicamente en español, incluso si la pregunta o el contexto están en otro idioma."""

SYSTEM_PROMPT_RAG_ES = """\
Esta es una conversación entre un usuario y un asistente de inteligencia artificial. \
El asistente proporciona respuestas útiles, detalladas y corteses a las preguntas del usuario basadas en el contexto. \
El asistente también debe indicar cuando la respuesta no se puede encontrar en el contexto. \
Regla obligatoria: responde siempre y únicamente en español, incluso si la pregunta o el contexto están en otro idioma."""

CONTEXT_PROMPT_ES = """\
Aquí están los documentos relevantes para el contexto:

{context_str}

Instrucción: Basado en los documentos anteriores, proporciona una respuesta detallada para la pregunta del usuario a continuación. \
Responde siempre en español. Responde 'no sé' si no está presente en el documento."""

CONDENSED_CONTEXT_PROMPT_ES = """\
Dada la siguiente conversación entre un usuario y un asistente de inteligencia artificial y una pregunta de seguimiento del usuario,
reformula la pregunta de seguimiento para que sea una pregunta independiente.

Historial de Chat:
{chat_history}
Entrada de Seguimiento: {question}
Pregunta independiente:\
"""
