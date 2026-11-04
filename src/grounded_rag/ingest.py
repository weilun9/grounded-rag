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
_MD_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+)$", re.MULTILINE)
_PEP_HEADER = re.compile(r"^(?P<key>[A-Z][\w-]*):\s*(?P<value>.*)$")
# Sections whose content is NOT what the document ends up proposing ("Rejected Ideas",
# "Rejected alternative proposals", "Rejected/Postponed Proposals", ...). Matched against the
# whole heading path, so a subsection like "Special-casing conditional statements" under
# "Rejected alternative proposals" is caught too.
_REJECTED_SECTION = re.compile(r"\b(rejected|abandoned|postponed|withdrawn)\b", re.IGNORECASE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source: str
    position: int
    # True when the chunk sits under a "Rejected ..." heading: the passage describes an idea the
    # document turned down. Defaults to False so indexes written before this field still load.
    rejected: bool = False

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


def _heading_marks(body: str) -> list[tuple[int, int, str, int]]:
    """[(start, end, title, level)] for every heading. RST levels follow the order in which underline
    characters first appear (that is how reStructuredText itself assigns them); Markdown uses the # count."""
    marks: list[tuple[int, int, str, int]] = []
    levels: dict[str, int] = {}
    for m in _RST_HEADING.finditer(body):
        level = levels.setdefault(m["under"][0], len(levels))
        marks.append((m.start(), m.end(), m["title"].strip(), level))
    if not marks:
        for m in _MD_HEADING.finditer(body):
            level = len(m.group(0)) - len(m.group(0).lstrip("#")) - 1
            marks.append((m.start(), m.end(), m["title"].strip(), level))
    return marks


def split_sections_with_path(body: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, text)] in document order, e.g. (["Rejected Ideas", "Using as"], "...").
    Text before the first heading has the path ["Preamble"]."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []  # (level, title) of the enclosing headings
    prev_end, prev_path = 0, ["Preamble"]
    for start, end, title, level in _heading_marks(body):
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
    return [(path[-1], text) for path, text in split_sections_with_path(body)]


def is_rejected_section(path: list[str]) -> bool:
    return any(_REJECTED_SECTION.search(h) for h in path)


def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []
    for doc in docs:
        position = 0
        for path, text in split_sections_with_path(doc.body):
            section, rejected = path[-1], is_rejected_section(path)
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
                        rejected=rejected,
                    )
                )
                position += 1
    return chunks
