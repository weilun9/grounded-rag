"""Load documents and split them into section-aware chunks.

Every chunk keeps the document title and the section heading it came from. That
header is prepended to the text that gets embedded ("contextual chunk headers"),
so a passage like "The default is False." still retrieves for questions that
name the feature, even though the passage itself never does.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".rst", ".md", ".txt", ".pdf"}

# reStructuredText heading: a text line followed by a line of one repeated punctuation char.
_RST_HEADING = re.compile(r"^(?P<title>[^\n]+)\n(?P<under>([=\-~^\"'`#*+])\3{2,})[ \t]*$", re.MULTILINE)
_MD_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+)$", re.MULTILINE)
_PEP_HEADER = re.compile(r"^(?P<key>[A-Z][\w-]*):\s*(?P<value>.*)$")
# Sections that describe ideas the document did NOT adopt. Matched against the heading and all its ancestors,
# so "Rejected Ideas > Using `as`" is caught as well. Two words are narrowed to avoid design-term hits:
# "Alternative(s)" only as the heading's first word (PEP 318's "Syntax Alternatives" surveys the options and
# explains the adopted @ syntax), and "Deferred" only as "Deferred ideas/proposals/..." (PEP 703's
# "Deferred Reference Counting" is part of the design).
_REJECTED_HEADING = re.compile(
    r"\b(rejected|abandoned)\b|^alternatives?\b|\bdeferred\s+(ideas?|proposals?|features?|items?)\b|^deferred$",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source: str
    position: int
    # "rejected" for chunks under a rejected/alternative/abandoned/deferred heading, else "normative"
    status: str = "normative"

    @property
    def embed_text(self) -> str:
        return f"{self.title} > {self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    source: str


def _parse_pep_header(text: str) -> tuple[dict[str, str], str]:
    """Split the RFC-822 style header block off a PEP. Returns ({}, text) if there is none."""
    lines = text.splitlines()
    header: dict[str, str] = {}
    last_key = None
    for i, line in enumerate(lines):
        if not line.strip():
            if header:
                return header, "\n".join(lines[i + 1 :])
            return {}, text
        m = _PEP_HEADER.match(line)
        if m:
            last_key = m["key"]
            header[last_key] = m["value"].strip()
        elif line.startswith((" ", "\t")) and last_key:
            header[last_key] += " " + line.strip()
        else:
            return {}, text
    return header, ""


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def load_document(path: Path) -> Document:
    text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="replace")
    header, body = _parse_pep_header(text)
    if "PEP" in header:
        num = int(header["PEP"])
        doc_id = f"PEP {num}"
        title = f"PEP {num} – {header.get('Title', '')}".strip(" –")
    else:
        doc_id = path.stem
        title = path.stem.replace("_", " ").replace("-", " ")
        body = text
    return Document(doc_id=doc_id, title=title, body=body, source=path.name)


def load_corpus(corpus_dir: Path) -> list[Document]:
    paths = sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
    return [load_document(p) for p in paths]


def _headings(body: str) -> list[tuple[int, int, int, str]]:
    """Return [(start, end, level, title)]. RST levels follow the order underline characters first appear in."""
    rst = list(_RST_HEADING.finditer(body))
    if rst:
        order: list[str] = []
        marks = []
        for m in rst:
            char = m["under"][0]
            if char not in order:
                order.append(char)
            marks.append((m.start(), m.end(), order.index(char), m["title"].strip()))
        return marks
    return [(m.start(), m.end(), len(m["hashes"]), m["title"].strip()) for m in _MD_HEADING.finditer(body)]


def split_section_paths(body: str) -> list[tuple[list[str], str]]:
    """Return [(heading path, text)] in document order, e.g. (["Rejected Ideas", "Alternative spellings"], ...).

    Text before the first heading gets the path ["Preamble"].
    """
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []  # (level, title) of the open headings, outermost first
    prev_end, prev_path = 0, ["Preamble"]
    for start, end, level, title in _headings(body):
        chunk = body[prev_end:start].strip()
        if chunk:
            sections.append((prev_path, chunk))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        prev_end, prev_path = end, [t for _, t in stack]
    tail = body[prev_end:].strip()
    if tail:
        sections.append((prev_path, tail))
    return sections


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return [(heading, text)] in document order. Text before the first heading is 'Preamble'."""
    return [(path[-1], text) for path, text in split_section_paths(body)]


def section_status(path: list[str]) -> str:
    return "rejected" if any(_REJECTED_HEADING.search(h) for h in path) else "normative"


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for doc in docs:
        position = 0
        for path, text in split_section_paths(doc.body):
            section, status = path[-1], section_status(path)
            for piece in splitter.split_text(text):
                if len(piece.strip()) < 40:  # stray underline fragments, lone references
                    continue
                digest = hashlib.sha1(f"{doc.doc_id}|{position}|{piece}".encode()).hexdigest()[:12]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id.replace(' ', '')}-{position:03d}-{digest}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        section=section,
                        text=piece.strip(),
                        source=doc.source,
                        position=position,
                        status=status,
                    )
                )
                position += 1
    return chunks
