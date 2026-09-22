"""MedQuAD ingestion helpers.

MedQuAD stores biomedical question/answer pairs in XML documents. This module
extracts only QAPair records with non-empty Question and Answer elements and
normalizes them into the instruction-data shape used by the project.
"""

from pathlib import Path
from xml.etree import ElementTree


def _text(element):
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def iter_medquad_records(source_root):
    """Yield ``(question, answer, source)`` records from a MedQuAD tree."""
    source_root = Path(source_root)
    medquad_root = source_root / "MedQuAD"
    if not medquad_root.is_dir():
        return

    for path in sorted(medquad_root.rglob("*.xml")):
        try:
            root = ElementTree.parse(path).getroot()
        except (ElementTree.ParseError, OSError):
            continue

        source = str(path.relative_to(source_root)).replace("\\", "/")
        for qa_pair in root.findall(".//QAPair"):
            question = _text(qa_pair.find("Question"))
            answer = _text(qa_pair.find("Answer"))
            if question and answer:
                yield question, answer, source
