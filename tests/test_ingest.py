from grounded_rag.config import ROOT
from grounded_rag.ingest import (
    Document,
    _parse_pep_header,
    chunk_documents,
    is_rejected_section,
    load_corpus,
    load_document,
    split_sections,
    split_sections_with_path,
)

from .conftest import DOCS


def test_pep_header_is_parsed_and_stripped():
    text = "PEP: 572\nTitle: Assignment Expressions\nAuthor: A,\n        B\n\nAbstract\n========\n\nBody text."
    header, body = _parse_pep_header(text)
    assert header["PEP"] == "572"
    assert header["Author"] == "A, B"  # continuation line folded
    assert body.startswith("Abstract")


def test_text_without_header_is_left_alone():
    header, body = _parse_pep_header("Just a plain document.\nSecond line.")
    assert header == {} and body.startswith("Just a plain")


def test_rst_sections_keep_their_headings():
    sections = split_sections(DOCS[1].body)
    assert [s for s, _ in sections] == ["Indentation", "Maximum Line Length"]
    assert "4 spaces" in sections[0][1]


def test_markdown_headings_are_used_when_there_are_no_rst_headings():
    sections = split_sections("intro text here\n\n# Setup\ninstall it\n\n## Usage\nrun it")
    assert [s for s, _ in sections] == ["Preamble", "Setup", "Usage"]


def test_chunks_carry_title_and_section_into_embed_text(chunks):
    c = next(c for c in chunks if "79 characters" in c.text)
    assert c.doc_id == "PEP 8" and c.section == "Maximum Line Length"
    assert c.embed_text.startswith("PEP 8 – Style Guide for Python Code > Maximum Line Length\n")


def test_chunk_ids_are_unique_and_stable():
    a = chunk_documents(DOCS, chunk_size=300, chunk_overlap=0)
    b = chunk_documents(DOCS, chunk_size=300, chunk_overlap=0)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


# ---- chunking ------------------------------------------------------------------------

NESTED = """Rationale
=========

Why we want this feature at all, explained at some length.

Rejected Ideas
==============

Using ``as``
------------

The spelling ``if x as y:`` was considered and turned down because it is ambiguous.

Specification
=============

The final syntax is ``NAME := expr`` and that is what implementations must accept.
"""


def test_heading_path_tracks_rst_nesting():
    paths = [p for p, _ in split_sections_with_path(NESTED)]
    assert paths == [["Rationale"], ["Rejected Ideas", "Using ``as``"], ["Specification"]]


def test_markdown_heading_path_uses_hash_depth():
    paths = [p for p, _ in split_sections_with_path("# A\ntext a\n## B\ntext b\n# C\ntext c")]
    assert paths == [["A"], ["A", "B"], ["C"]]


def test_rejected_sections_are_detected_through_the_parent_heading():
    assert is_rejected_section(["Rejected alternative proposals", "Special-casing conditional statements"])
    assert is_rejected_section(["Rejected/Postponed Proposals"])
    assert not is_rejected_section(["Specification", "Alternatives considered for the grammar"])


def test_chunks_under_rejected_headings_are_flagged():
    chunks = chunk_documents([Document("PEP 1", "PEP 1 – Test", NESTED, "x.rst")], chunk_size=300, chunk_overlap=0)
    flagged = {c.section: c.rejected for c in chunks}
    assert flagged == {"Rationale": False, "Using ``as``": True, "Specification": False}


def test_chunks_respect_the_size_limit_and_overlap():
    body = "Section\n=======\n\n" + " ".join(f"word{i}" for i in range(400))
    chunks = chunk_documents([Document("D", "D", body, "d.txt")], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    # consecutive pieces share text because of the overlap
    assert chunks[0].text.split()[-1] in chunks[1].text
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_tiny_fragments_are_dropped():
    body = "Real\n====\n\nA section with enough text to be worth keeping as a passage.\n\nTiny\n====\n\nok"
    chunks = chunk_documents([Document("D", "D", body, "d.txt")], chunk_size=300, chunk_overlap=0)
    assert [c.section for c in chunks] == ["Real"]


def test_plain_text_file_falls_back_to_file_name(tmp_path):
    f = tmp_path / "my_notes-v2.md"
    f.write_text("# Intro\nhello there", encoding="utf-8")
    doc = load_document(f)
    assert doc.doc_id == "my_notes-v2" and doc.title == "my notes v2" and doc.body.startswith("# Intro")


def test_demo_corpus_matches_the_numbers_in_the_readme():
    """README states 29 documents / 1,571 chunks with the default chunk settings (1000 / 150)."""
    docs = load_corpus(ROOT / "data" / "corpus")
    chunks = chunk_documents(docs, 1000, 150)
    assert len(docs) == 29
    assert len(chunks) == 1571
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    pep572 = [c for c in chunks if c.doc_id == "PEP 572" and "as match" in c.text]
    assert pep572 and all(c.rejected for c in pep572)
