from typing import List
from openai import OpenAI
from core_logic.env_config import get_required_env

def get_llm_model():
    return get_required_env("LLM_MODEL")

def get_prompt_strategy():
    return get_required_env("PROMPT_STRATEGY").lower()

def get_response_language():
    return get_required_env("RESPONSE_LANGUAGE").lower()

def get_openai_client():
    """Create OpenAI client configured for custom API"""
    base_url = get_required_env("OPENAI_BASE_URL")
    # API key is optional for some custom deployments, use dummy value if not required
    api_key = "not-needed"  # Your API doesn't require authentication

    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )

def construct_prompt_fast(query: str, context_docs: List[str]) -> str:
    """Fast strategy: minimal instructions for quick responses with simple documents"""
    context_xml = "\n\n".join(
        f"<context>{doc}</context>"
        for doc in context_docs
    )

    prompt = f"""Answer the question using the provided context. If the answer isn't in the context, say "I don't know."

<contexts>
{context_xml}
</contexts>

Question: {query}

Answer:"""

    return prompt

def construct_prompt_balanced(query: str, context_docs: List[str]) -> str:
    """Balanced strategy: good default with clear grounding and anti-hallucination measures"""
    context_xml = "\n\n".join(
        f"<context id='{i}'>\n{doc}\n</context>"
        for i, doc in enumerate(context_docs, 1)
    )

    prompt = f"""Answer the question using ONLY the information in the provided context below.

Rules:
- Use ONLY the provided context to answer
- If the answer is not in the context, respond with: "I don't have enough information to answer this question."
- Do not use prior knowledge or make assumptions
- Be concise and accurate

<contexts>
{context_xml}
</contexts>

<question>
{query}
</question>

Answer:"""

    return prompt

def construct_prompt_precise(query: str, context_docs: List[str]) -> str:
    """Precise strategy: strong anti-hallucination with step-by-step reasoning"""
    context_xml = "\n\n".join(
        f"<context id='{i}'>\n{doc}\n</context>"
        for i, doc in enumerate(context_docs, 1)
    )

    prompt = f"""Answer the question using ONLY the provided context below.

<instruction>
1. Read the context documents carefully
2. If the answer is found in the context, provide a clear answer and mention which context ID(s) you used
3. If the answer is NOT in the context, respond EXACTLY with: "I don't have enough information in the provided context to answer this question."
4. Do NOT use your prior knowledge or make assumptions beyond what's explicitly stated
5. If you're uncertain, err on the side of saying you don't know
</instruction>

<contexts>
{context_xml}
</contexts>

<question>
{query}
</question>

Answer:"""

    return prompt

def construct_prompt_comprehensive(query: str, context_docs: List[str]) -> str:
    """Comprehensive strategy: maximum accuracy with chain-of-thought and self-critique"""
    context_xml = "\n\n".join(
        f"<context id='{i}'>\n{doc}\n</context>"
        for i, doc in enumerate(context_docs, 1)
    )

    prompt = f"""Answer the question using ONLY the provided context below.

<instruction>
Follow this step-by-step process:

Step 1: Identify which context documents contain relevant information for the question
Step 2: Extract the specific facts or statements that relate to the question
Step 3: Formulate your answer using ONLY the extracted information
Step 4: Review your answer to ensure it doesn't contain information not in the context
Step 5: Cite which context ID(s) you used

If at any step you find insufficient information in the context:
- Respond EXACTLY with: "I don't have enough information in the provided context to answer this question."
- Do NOT use your prior knowledge
- Do NOT make assumptions or inferences beyond what's explicitly stated

If you're uncertain or the information is ambiguous, always err on the side of saying you don't know.
</instruction>

<contexts>
{context_xml}
</contexts>

<question>
{query}
</question>

Let's work through this step by step:"""

    return prompt

# French prompts
def construct_prompt_fast_fr(query: str, context_docs: List[str]) -> str:
    """Stratégie rapide : instructions minimales pour des réponses rapides"""
    context_xml = "\n\n".join(
        f"<contexte>{doc}</contexte>"
        for doc in context_docs
    )

    prompt = f"""Réponds à la question en utilisant le contexte fourni. Si la réponse n'est pas dans le contexte, dis "Je ne sais pas."

<contextes>
{context_xml}
</contextes>

Question : {query}

Réponse :"""

    return prompt

def construct_prompt_balanced_fr(query: str, context_docs: List[str]) -> str:
    """Stratégie équilibrée : bon équilibre entre précision et vitesse avec mesures anti-hallucination"""
    context_xml = "\n\n".join(
        f"<contexte id='{i}'>\n{doc}\n</contexte>"
        for i, doc in enumerate(context_docs, 1)
    )

    prompt = f"""Réponds à la question en utilisant UNIQUEMENT les informations du contexte fourni ci-dessous.

Règles :
- Utilise UNIQUEMENT le contexte fourni pour répondre
- Si la réponse n'est pas dans le contexte, réponds : "Je n'ai pas assez d'informations pour répondre à cette question."
- N'utilise pas de connaissances antérieures et ne fais pas d'hypothèses
- Sois concis et précis

<contextes>
{context_xml}
</contextes>

<question>
{query}
</question>

Réponse :"""

    return prompt

def construct_prompt_precise_fr(query: str, context_docs: List[str]) -> str:
    """Stratégie précise : forte anti-hallucination avec raisonnement étape par étape"""
    context_xml = "\n\n".join(
        f"<contexte id='{i}'>\n{doc}\n</contexte>"
        for i, doc in enumerate(context_docs, 1)
    )

    prompt = f"""Réponds à la question en utilisant UNIQUEMENT le contexte fourni ci-dessous.

<instruction>
1. Lis attentivement les documents de contexte
2. Si la réponse se trouve dans le contexte, fournis une réponse claire et mentionne quel(s) ID(s) de contexte tu as utilisé
3. Si la réponse n'est PAS dans le contexte, réponds EXACTEMENT : "Je n'ai pas assez d'informations dans le contexte fourni pour répondre à cette question."
4. N'utilise PAS tes connaissances antérieures et ne fais pas d'hypothèses au-delà de ce qui est explicitement indiqué
5. En cas de doute, préfère dire que tu ne sais pas
</instruction>

<contextes>
{context_xml}
</contextes>

<question>
{query}
</question>

Réponse :"""

    return prompt

def construct_prompt_comprehensive_fr(query: str, context_docs: List[str]) -> str:
    """Stratégie complète : précision maximale avec chaîne de pensée et auto-critique"""
    context_xml = "\n\n".join(
        f"<contexte id='{i}'>\n{doc}\n</contexte>"
        for i, doc in enumerate(context_docs, 1)
    )

    prompt = f"""Réponds à la question en utilisant UNIQUEMENT le contexte fourni ci-dessous.

<instruction>
Suis ce processus étape par étape :

Étape 1 : Identifie quels documents de contexte contiennent des informations pertinentes pour la question
Étape 2 : Extrais les faits ou déclarations spécifiques liés à la question
Étape 3 : Formule ta réponse en utilisant UNIQUEMENT les informations extraites
Étape 4 : Vérifie que ta réponse ne contient pas d'informations absentes du contexte
Étape 5 : Cite quel(s) ID(s) de contexte tu as utilisé

Si à n'importe quelle étape tu trouves des informations insuffisantes dans le contexte :
- Réponds EXACTEMENT : "Je n'ai pas assez d'informations dans le contexte fourni pour répondre à cette question."
- N'utilise PAS tes connaissances antérieures
- Ne fais PAS d'hypothèses ou d'inférences au-delà de ce qui est explicitement indiqué

En cas de doute ou d'ambiguïté, préfère toujours dire que tu ne sais pas.
</instruction>

<contextes>
{context_xml}
</contextes>

<question>
{query}
</question>

Travaillons étape par étape :"""

    return prompt

def construct_prompt(query: str, context_docs: List[str]) -> str:
    """Route to appropriate prompt strategy based on config and language"""
    strategy = get_prompt_strategy()
    language = get_response_language()

    # French prompts
    if language == "fr":
        if strategy == "fast":
            return construct_prompt_fast_fr(query, context_docs)
        elif strategy == "precise":
            return construct_prompt_precise_fr(query, context_docs)
        elif strategy == "comprehensive":
            return construct_prompt_comprehensive_fr(query, context_docs)
        else:  # balanced (default)
            return construct_prompt_balanced_fr(query, context_docs)

    # English prompts (default)
    else:
        if strategy == "fast":
            return construct_prompt_fast(query, context_docs)
        elif strategy == "precise":
            return construct_prompt_precise(query, context_docs)
        elif strategy == "comprehensive":
            return construct_prompt_comprehensive(query, context_docs)
        else:  # balanced (default)
            return construct_prompt_balanced(query, context_docs)

def generate_response(query: str, context: str) -> str:
    client = get_openai_client()
    model = get_llm_model()

    # Construct prompt with context
    if context:
        prompt = construct_prompt(query, [context])
    else:
        prompt = f"Question: {query}\n\nPlease answer the question. If you don't have enough information, say so."

    # Generate response using OpenAI Chat Completions API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content
