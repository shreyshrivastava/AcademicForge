"""Explicit field constraints and transparent expansion of ambiguous search terms."""

import re

FIELD_PRESETS = {
    'Computer science & AI': 'computing',
    'Music & arts': 'music',
    'Medicine & health': 'health',
    'Energy & environment': 'energy',
}
ALIASES = {
    'all fields': 'all', 'all': 'all',
    'computer science': 'computing', 'computer science & ai': 'computing',
    'computing': 'computing', 'ai': 'computing', 'machine learning': 'computing',
    'computer science and ai': 'computing', 'software engineering': 'computing', 'artificial intelligence': 'computing',
    'music': 'music', 'music & arts': 'music', 'arts': 'music',
    'health': 'health', 'medicine': 'health', 'medicine & health': 'health',
    'energy': 'energy', 'energy & environment': 'energy', 'environment': 'energy',
}
FIELD_TERMS = {
    'computing': {'algorithm', 'algorithms', 'computational', 'computer', 'computing', 'software',
                  'neural', 'nlp', 'llm', 'llms', 'retrieval', 'programming', 'transformer', 'transformers'},
    'music': {'music', 'musical', 'musician', 'musicians', 'melody', 'melodies', 'raga',
              'rhythm', 'rhythmic', 'tonal', 'raag', 'singer', 'singing', 'painting', 'sculpture', 'artistic'},
    'health': {'medical', 'medicine', 'health', 'healthcare', 'clinical', 'patient', 'patients',
               'disease', 'diseases', 'therapy', 'treatment', 'nutrition', 'obesity', 'cancer', 'biology'},
    'energy': {'energy', 'environment', 'environmental', 'climate', 'solar', 'wind', 'renewable',
               'electricity', 'photovoltaic', 'emissions', 'grid', 'carbon'},
}
RAG = re.compile(r'\brag\b', re.IGNORECASE)
RAGAS = re.compile(r'\bragas\b|\brag\s+(?:and\s+)?as\b(?!\s+(?:a|an|the)\b)', re.IGNORECASE)


def field_kind(domain):
    return ALIASES.get(domain.strip().casefold(), 'custom' if domain.strip() else 'all')


def contextualize_query(query, domain=''):
    """Only expand the computing meanings when the user explicitly chooses that field."""
    query = query.strip()
    kind = field_kind(domain)
    if kind == 'all':
        return query
    if kind == 'computing':
        if RAGAS.search(query):
            expanded = RAGAS.sub('RAGAS', query)
            return expanded + ' retrieval augmented generation evaluation'
        if RAG.search(query):
            expanded = RAG.sub('retrieval augmented generation', query)
            return expanded
        return query + ' computer science'
    suffix = {'music': 'music arts', 'health': 'medicine health', 'energy': 'energy environment'}.get(kind, domain.strip())
    return query + ' ' + suffix


def _paper_text(paper):
    metadata = paper.get('metadata', {}) or {}
    return ' '.join([
        paper.get('title', ''), paper.get('abstract', ''),
        ' '.join(str(item) for item in metadata.get('fields_of_study', [])),
        ' '.join(str(item) for item in metadata.get('categories', [])),
        str(metadata.get('primary_category', '')),
    ]).casefold()


def _supports_rag_intent(text):
    if re.search(r'\bretrieval[\s-]+augmented\b', text):
        return True
    terms = set(re.findall(r'[a-z]+', text))
    return bool({'rag', 'ragas'} & terms) and bool({'retrieval', 'llm', 'llms', 'nlp'} & terms)


def matches_field(paper, query, domain=''):
    kind = field_kind(domain)
    if kind == 'all':
        return True
    text = _paper_text(paper)
    terms = set(re.findall(r'[a-z0-9]+', text))
    if kind == 'custom':
        requested = set(re.findall(r'[a-z0-9]+', domain.casefold())) - {'and', 'of', 'the', 'in'}
        return bool(requested) and requested.issubset(terms)
    if kind == 'computing' and (RAG.search(query) or RAGAS.search(query)):
        # An acoustic raga classifier can be computer science too; it still is not RAG retrieval.
        return _supports_rag_intent(text)
    if kind == 'computing' and (re.search(r'\bcs\.[a-z]+\b', text) or 'computer science' in text):
        return True
    return bool(terms & FIELD_TERMS[kind])
