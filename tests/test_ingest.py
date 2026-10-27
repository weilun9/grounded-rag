from grounded_rag.ingest import _parse_pep_header, chunk_documents, split_sections

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
