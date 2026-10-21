"""Small extractive evidence analysis; no learned models or external services.

Sentences and reported measurements are extracted from supplied abstracts.
Build recommendations are explicitly proposals, not claims from those papers.
"""

from collections import Counter
import math
import re

from backend.retrieval.bm25 import tokenize

STOP = set("a an the and or of to in on for with by from as is are was were be been this that these those "
           "we our their it its they have has had not also can could would should will may such using used "
           "use paper study work approach proposed propose presents present show shows results new based "
           "through between than which into both each more most other across however demonstrate".split())
PATTERNS = {
    "method": r"\b(?:propos\w*|introduc\w*|develop\w*|framework|algorithm|architecture|method|pipeline|train\w*)\b",
    "evaluation": r"\b(?:evaluat\w*|benchmark\w*|dataset\w*|experiment\w*|compar\w*|baseline\w*|test\w*)\b",
    "result": r"\b(?:outperform\w*|improv\w*|achiev\w*|reduc\w*|increase\w*|result\w*|findings)\b|\d+(?:\.\d+)?\s*%",
    "limitation": r"\b(?:limitation\w*|fail\w*|challenge\w*|trade.off|restrict\w*|despite|bias\w*|only|small.scale)\b",
}
METRICS = re.compile(r"\b(?:nDCG(?:@\d+)?|Recall(?:@\d+)?|Precision(?:@\d+)?|MRR(?:@\d+)?|MAP|"
                     r"F1(?:.score)?|accuracy|AUROC|AUC|RMSE|MAE|BLEU|ROUGE(?:.\w+)?|"
                     r"faithfulness|answer relevance|context precision|context recall|latency|throughput)\b", re.I)
DATASETS = re.compile(r"\b(?:SciFact|BEIR|MS MARCO|Natural Questions|SQuAD|HotpotQA|TriviaQA|"
                      r"ImageNet|CIFAR(?:.\d+)?|MIMIC(?:.\w+)?|PubMedQA|MedQA|GLUE|SuperGLUE)\b", re.I)


def terms(text):
    return [t for t in tokenize(text) if t not in STOP and len(t) > 2]


def sentences(text):
    text = re.sub(r"\s+", " ", text or "").strip()[:20000]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text) if s.strip()]


def ranked_sentences(text, query=""):
    items = sentences(text)
    frequency = Counter(terms(text))
    focus = set(terms(query))
    scored = []
    for i, sentence in enumerate(items):
        words = terms(sentence)
        if not words:
            continue
        score = sum(math.log1p(frequency[w]) for w in set(words)) / math.sqrt(len(words))
        score += 2 * len(focus.intersection(words)) / math.sqrt(len(words))
        score += sum(bool(re.search(pattern, sentence, re.I)) for pattern in PATTERNS.values())
        scored.append((score, i, sentence))
    return sorted(scored, key=lambda row: (-row[0], row[1]))


def condense(text, query="", max_words=90):
    ranked = ranked_sentences(text, query)
    if not ranked:
        return "No usable abstract is available."
    # Select fewer sentences than the source whenever possible, preserving source order.
    limit = min(3, max(1, math.ceil(len(sentences(text)) * 0.4)))
    chosen, words = [], 0
    for _, i, sentence in ranked:
        count = len(sentence.split())
        if chosen and words + count > max_words:
            continue
        chosen.append((i, sentence))
        words += count
        if len(chosen) >= limit or words >= max_words:
            break
    result = " ".join(s for _, s in sorted(chosen))
    if len(result.split()) > max_words:
        result = " ".join(result.split()[:max_words]) + "…"
    return result


def keyphrases(text, limit=5):
    words = tokenize(text)
    phrases = Counter()
    for size in (2, 3):
        for start in range(len(words) - size + 1):
            group = words[start:start + size]
            if all(w not in STOP and len(w) > 2 for w in group):
                phrases[' '.join(group)] += 1
    selected = []
    for phrase, _ in sorted(phrases.items(), key=lambda row: (-row[1] * len(row[0].split()), row[0])):
        if not any(phrase in previous or previous in phrase for previous in selected):
            selected.append(phrase)
        if len(selected) >= limit:
            break
    return selected


def analyze(paper, query=""):
    abstract = paper.get("abstract") or ""
    ranked = ranked_sentences(abstract, query)
    signals = {name: next((s for _, _, s in ranked if re.search(pattern, s, re.I)), None)
               for name, pattern in PATTERNS.items()}
    return {"title": paper.get("title", "Untitled"), "summary": condense(abstract, query),
            "signals": signals, "metrics": sorted(set(METRICS.findall(abstract))),
            "datasets": sorted(set(DATASETS.findall(abstract))),
            "concepts": keyphrases(paper.get("title", "") + '. ' + abstract),
            "source": paper.get("url") or paper.get("link") or "Not supplied"}


def paper_summary(paper):
    info = analyze(paper, paper.get("query", ""))
    lines = ["Extractive summary · selected from the abstract; no AI model.", "", "Core contribution", info['summary']]
    for name, label in (("method", "Method"), ("evaluation", "Evaluation"), ("result", "Reported result")):
        value = info['signals'][name]
        if value and value not in info['summary']:
            lines += ["", label, value]
    if info['metrics']:
        lines += ["", "Measures mentioned", ', '.join(info['metrics'])]
    if info['concepts']:
        lines += ["", "Key concepts", ', '.join(info['concepts'])]
    lines += ["", "Limits", info['signals']['limitation'] or "The abstract does not explicitly state limitations; check the full paper.",
              "", "Source", info['source']]
    return '\n'.join(lines)


def paper_guidance(paper):
    info = analyze(paper, paper.get("query", ""))
    method = info['signals']['method']
    evaluation = info['signals']['evaluation']
    return '\n\n'.join([
        "Evidence-based implementation checklist · no AI model; verify details in the full paper.",
        "What to reproduce\n" + (method or info['summary']),
        "Implementation steps\n" + '\n'.join([
            f"1. Use {info['title']} as the reference; identify its inputs, outputs and algorithm before implementation.",
            f"2. Locate the implementation details for: {', '.join(info['concepts'][:3]) or 'the stated method'}.",
            "3. Reproduce the reference configuration before changing one component at a time.",
        ]),
        "Evaluation evidence\n" + (evaluation or "No evaluation procedure is explicit in this abstract."),
        "Measures to locate\n" + (', '.join(info['metrics']) or "No named metric was identified in the abstract; select one appropriate to your task."),
        "Data to locate\n" + (', '.join(info['datasets']) or "No recognized dataset name was identified; verify the data and its access terms in the paper."),
        "Failure checks\n" + (info['signals']['limitation'] or "The abstract does not describe failure cases. Test robustness and reproducibility rather than assuming the method generalizes."),
        "Source\n" + info['source'],
    ])


def build_recipe(query, infos):
    text = (query + ' ' + ' '.join(i['title'] + ' ' + i['summary'] for i in infos)).lower()
    if re.search(r'\bretrieval\b|\brag(?:as)?\b', text):
        return ("Documents → chunks → retriever → retrieved evidence → answer generation → grounded-answer evaluation",
                ["Prepare a small document collection and a set of questions with supporting passages.",
                 "Implement a BM25 baseline; keep chunk boundaries and source identifiers reproducible.",
                 "Evaluate retrieval before adding an answer generator; compare one retrieval or chunking change at a time."],
                "Recall@10 and nDCG@10 for labeled retrieval; citation support and latency for generated answers")
    if re.search(r'\bforecast\w*\b|\btime series\b', text):
        return ("Time-indexed observations → chronological split → baseline forecast → candidate method → error analysis",
                ["Create chronological training and held-out periods without future-data leakage.",
                 "Start with a seasonal or last-value baseline before implementing the selected method.",
                 "Compare errors across forecast horizons and record runtime."], "MAE and RMSE on a held-out period")
    if re.search(r'\bclassif\w*\b|\bdiagnos\w*\b|\bdetect\w*\b', text):
        return ("Labeled examples → independent split → baseline classifier → candidate method → subgroup error analysis",
                ["Define labels and separate training and evaluation examples before preprocessing.",
                 "Reproduce a simple baseline before adapting the paper's method.",
                 "Measure class-specific errors and test performance on held-out groups."], "Precision, recall and F1, with per-class errors")
    return ("Task definition → reproducible inputs → baseline → selected paper method → held-out evaluation",
            [f"Define measurable inputs and outputs for: {query or 'the selected research question'}.",
             "Reproduce the method described in the first selected paper before extending it.",
             "Hold the evaluation data fixed and compare one implementation change at a time."],
            "A task-appropriate quality measure, runtime and documented failure cases")


def research_plan(payload):
    query = payload.get('query', '')
    infos = [analyze(p, query) for p in (payload.get('papers') or [])[:10]]
    if not infos:
        return "# Research Focus\nSelect evidence before building a plan."
    architecture, steps, suggested_metrics = build_recipe(query, infos)
    metrics = sorted({m for info in infos for m in info['metrics']})
    datasets = sorted({d for info in infos for d in info['datasets']})
    findings = []
    for index, info in enumerate(infos, 1):
        findings.append(f"- P{index} — {info['title']}: {info['summary']}")
        if info['metrics']:
            findings.append(f"- P{index} names these measures: {', '.join(info['metrics'])}.")
    gaps = []
    for name, description in (("evaluation", "evaluation procedure"), ("limitation", "limitations")):
        missing = [f"P{i}" for i, info in enumerate(infos, 1) if not info['signals'][name]]
        if missing:
            gaps.append(f"- {', '.join(missing)}: the abstract does not explicitly describe {description}; inspect the full text.")
    if not datasets:
        gaps.append("- No recognized dataset name was extracted. Confirm the original data, access terms and a reproducible split.")
    if not metrics:
        gaps.append("- No named evaluation metric was extracted. Define one before implementing a proposed extension.")
    if not gaps:
        gaps.append("- Abstracts mention evaluation and limitations; check whether their datasets and protocols are comparable before combining results.")
    references = '\n'.join(f"- P{i}: {info['title']} — {info['source']}" for i, info in enumerate(infos, 1))
    return '\n\n'.join([
        '# Research Focus',
        f"{query}\n\nSearch field: {payload.get('domain') or 'All fields'}\n\n"
        "Extractive evidence analysis and a proposed build checklist. No AI model is used. Claims below come from abstracts and are not independently verified.",
        '# Key Findings', '\n'.join(findings),
        '# Research Gaps', '\n'.join(gaps),
        '# Recommended Build',
        f"Recommended Architecture: Proposed starting point: {architecture}.\n\n"
        "Core Components:\n" + '\n'.join(f"- {step}" for step in steps) +
        f"\n- Reference method: {infos[0]['title']} (P1); verify its full implementation details.\n\n"
        "Deployment Considerations: Run the baseline on a small reproducible dataset, measure memory and runtime, then scale only after the comparison works.\n\n"
        "Evaluation Strategy: " + (f"Measures actually named in selected abstracts: {', '.join(metrics)}. " if metrics else '') +
        f"Proposed baseline checks: {suggested_metrics}. " +
        (f"Recognized datasets to inspect: {', '.join(datasets)}." if datasets else "Choose and document a dataset; none was recognized in these abstracts."),
        '# Builder Guidance', references,
    ])
