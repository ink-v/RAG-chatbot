def get_single_select_prompt(language: str):
    if language == "span":
        return single_select_prompt_es
    return single_select_prompt_en


single_select_prompt_en = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "ONE AND ONLY ONE choice that is most relevant to the query: '{query_str}'\n"
)

single_select_prompt_es = (
    "A continuación se dan algunas opciones, proporcionadas en una lista numerada "
    "(1 a {num_choices}), "
    "donde cada elemento de la lista corresponde a un resumen.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Regla obligatoria: razona y responde en español.\n"
    "Usando solo las opciones anteriores y no el conocimiento previo, devuelve "
    "UNA Y SOLO UNA opción que sea más relevante para la consulta: '{query_str}'\n"
)
