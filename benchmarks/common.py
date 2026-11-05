"""Shared helpers for benchmark and review scripts."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "scifact": {
        "label": "BEIR SciFact",
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "sha256": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        "documents": 5183,
        "queries": 300,
    },
    "nfcorpus": {
        "label": "BEIR NFCorpus",
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "sha256": "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        "documents": 3633,
        "queries": 323,
    },
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomically(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
        temporary_path = Path(file.name)
    temporary_path.replace(path)


def source_fingerprint(paths: list[str]) -> dict:
    return {name: file_sha256(ROOT / name) for name in paths}


def download(archive: Path, dataset: dict) -> None:
    import requests

    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".part")
    try:
        with requests.get(dataset["url"], timeout=(10, 60), stream=True) as response:
            response.raise_for_status()
            with partial.open("wb") as file:
                for chunk in response.iter_content(65536):
                    file.write(chunk)
        if file_sha256(partial) != dataset["sha256"]:
            raise ValueError(f"{dataset['label']} archive checksum does not match the recorded version")
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)


def read_beir(archive: Path, dataset_id: str, split: str = "test"):
    """Read a checksum-verified official BEIR split: raw corpus rows, queries, positive qrels."""
    dataset = DATASETS[dataset_id]
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != dataset["sha256"]:
        raise ValueError(f"{dataset['label']} archive checksum does not match the recorded version")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive_file:
        prefix = f"{dataset_id}/"
        corpus = [json.loads(line) for line in archive_file.read(f"{prefix}corpus.jsonl").splitlines()]
        queries = {
            str(query["_id"]): query["text"]
            for query in (json.loads(line) for line in archive_file.read(f"{prefix}queries.jsonl").splitlines())
        }
        qrels = {}
        rows = csv.DictReader(
            io.StringIO(archive_file.read(f"{prefix}qrels/{split}.tsv").decode()), delimiter="\t"
        )
        for row in rows:
            if int(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    if len(corpus) != dataset["documents"] or not qrels:
        raise ValueError(f"Unexpected {dataset['label']} corpus or empty qrels")
    ids = {str(document["_id"]) for document in corpus}
    if len(ids) != len(corpus) or not set(qrels).issubset(queries):
        raise ValueError(f"{dataset['label']} identifiers failed validation")
    if not all(relevant.issubset(ids) for relevant in qrels.values()):
        raise ValueError("Judgments reference documents outside the corpus")
    return corpus, queries, qrels
