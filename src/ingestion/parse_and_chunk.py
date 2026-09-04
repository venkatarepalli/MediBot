"""
Parse every PDF/Markdown in data/mediassist_data/<collection>/ with Docling's
structural parser, then chunk hierarchically with HybridChunker.
"""
import re
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc.labels import DocItemLabel

from src.rbac.access_config import get_access_roles, COLLECTIONS

DATA_ROOT = Path("data/mediassist_data")

# Several source documents use a consistent lettered top-level heading
# convention ("A. Type 2 Diabetes Mellitus", "B. Infusion Pump - DriveFlow
# IP-200"). Docling's automatic heading-nesting sometimes treats these as
# SIBLINGS of their own subsections rather than true parents - confirmed
# empirically: both Diabetes's and Hypertension's "Pharmacological
# management" subsections came back with identical, undistinguishable
# headings/text otherwise. Tracking this convention ourselves, across
# chunks in document order, fixes that regardless of what Docling's
# chunker considers "nested."
TOP_SECTION_PATTERN = re.compile(r"^[A-Z]\.\s+\S")


def _chunk_type(chunk) -> str:
    labels = {item.label for item in chunk.meta.doc_items}
    if DocItemLabel.TABLE in labels:
        return "table"
    if DocItemLabel.SECTION_HEADER in labels or DocItemLabel.TITLE in labels:
        return "heading"
    if DocItemLabel.CODE in labels:
        return "code"
    return "text"


def parse_and_chunk_file(filepath: Path, collection: str) -> list[dict]:
    converter = DocumentConverter()
    result = converter.convert(str(filepath))
    dl_doc = result.document

    chunker = HybridChunker()
    access_roles = get_access_roles(collection)

    records = []
    current_top_section = None

    for chunk in chunker.chunk(dl_doc):
        embedded_text = chunker.contextualize(chunk=chunk)
        nearest_heading = chunk.meta.headings[-1] if chunk.meta.headings else filepath.stem

        for heading in chunk.meta.headings:
            if TOP_SECTION_PATTERN.match(heading):
                current_top_section = heading
        if TOP_SECTION_PATTERN.match(nearest_heading):
            current_top_section = nearest_heading

        if current_top_section and current_top_section != nearest_heading:
            section_title = f"{current_top_section} > {nearest_heading}"
            embedded_text = f"{current_top_section}\n{embedded_text}"
        else:
            section_title = nearest_heading

        records.append({
            "text": embedded_text,
            "raw_text": chunk.text,
            "source_document": filepath.name,
            "collection": collection,
            "access_roles": access_roles,
            "section_title": section_title,
            "chunk_type": _chunk_type(chunk),
        })
    return records


def parse_and_chunk_collection(collection: str) -> list[dict]:
    folder = DATA_ROOT / collection
    all_records = []
    for filepath in sorted(folder.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".md"):
            continue
        print(f"  parsing {filepath.name} ...")
        records = parse_and_chunk_file(filepath, collection)
        print(f"    -> {len(records)} chunks")
        all_records.extend(records)
    return all_records


def parse_and_chunk_all() -> list[dict]:
    all_records = []
    for collection in COLLECTIONS:
        print(f"Collection: {collection}")
        all_records.extend(parse_and_chunk_collection(collection))
    return all_records