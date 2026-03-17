from llama_index.core import PromptTemplate


def get_query_gen_prompt(language: str):
    if language == "span":
        return query_gen_prompt_es
    return query_gen_prompt_en


query_gen_prompt_es = PromptTemplate(
    "Eres un generador de consultas de búsqueda hábil, dedicado a proporcionar consultas de búsqueda precisas y relevantes que sean concisas, específicas y sin ambigüedades.\n"
    "Regla obligatoria: genera las consultas siempre y únicamente en español.\n"
    "Genera {num_queries} consultas de búsqueda únicas y diversas, una en cada línea, relacionadas con la siguiente consulta de entrada:\n"
    "### Consulta Original: {query}\n"
    "### Por favor proporciona consultas de búsqueda que sean:\n"
    "- Relevantes a la consulta original\n"
    "- Bien definidas y específicas\n"
    "- Libres de ambigüedad y vaguedad\n"
    "- Útiles para recuperar resultados de búsqueda precisos y relevantes\n"
    "### Consultas Generadas:\n"
)

query_gen_prompt_en = PromptTemplate(
    "You are a skilled search query generator, dedicated to providing accurate and relevant search queries that are concise, specific, and unambiguous.\n"
    "Generate {num_queries} unique and diverse search queries, one on each line, related to the following input query:\n"
    "### Original Query: {query}\n"
    "### Please provide search queries that are:\n"
    "- Relevant to the original query\n"
    "- Well-defined and specific\n"
    "- Free of ambiguity and vagueness\n"
    "- Useful for retrieving accurate and relevant search results\n"
    "### Generated Queries:\n"
)
