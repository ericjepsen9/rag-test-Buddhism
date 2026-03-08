from typing import Dict, Any
from rag_runtime_config import PRODUCT_ALIASES, PROJECT_ALIASES, BUDDHIST_TERMS, CONCEPT_TERMS
from search_utils import detect_terms, uniq, split_multi_question


def rewrite_query(question: str) -> Dict[str, Any]:
    q = (question or "").strip()
    products = detect_terms(q, PRODUCT_ALIASES)
    projects = detect_terms(q, PROJECT_ALIASES)

    buddhist = [x for x in BUDDHIST_TERMS if x in q]
    concepts = [x for x in CONCEPT_TERMS if x in q]

    expanded_terms = []
    for pid in products:
        expanded_terms.extend(PRODUCT_ALIASES.get(pid, [])[:4])
    for pj in projects:
        expanded_terms.extend(PROJECT_ALIASES.get(pj, [])[:3])
    expanded_terms.extend(buddhist)
    expanded_terms.extend(concepts)

    sub_questions = split_multi_question(q)
    expanded_query = " ".join(uniq([q] + expanded_terms))

    return {
        "original": q,
        "expanded": expanded_query,
        "products": products,
        "projects": projects,
        "buddhist_terms": uniq(buddhist),
        "concepts": uniq(concepts),
        "sub_questions": sub_questions,
    }
