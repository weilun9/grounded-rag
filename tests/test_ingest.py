import pytest

from grounded_rag.ingest import _parse_pep_header, chunk_documents, section_status, split_section_paths, split_sections

from .conftest import DOCS, REJECTED_DOC


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


def test_section_paths_track_parent_headings():
    paths = [path for path, _ in split_section_paths(REJECTED_DOC.body)]
    assert paths == [["Specification"], ["Rejected Ideas", "Using ``as`` in conditions"],
                     ["Deferred Reference Counting"]]


def test_markdown_heading_levels_nest():
    paths = [p for p, _ in split_section_paths("# Rejected Ideas\n\n## Use as\ntext\n\n# Spec\nmore text")]
    assert paths == [["Rejected Ideas", "Use as"], ["Spec"]]


@pytest.mark.parametrize(("path", "status"), [
    (["Rejected Ideas", "Using as"], "rejected"),
    (["Rejected alternative proposals"], "rejected"),
    (["Rationale", "Alternative spellings"], "rejected"),
    (["Abandoned approaches"], "rejected"),
    (["Deferred Ideas"], "rejected"),
    (["Specification", "Deferred Reference Counting"], "normative"),  # PEP 703 design term
    (["Specification", "Exceptional cases"], "normative"),
])
def test_section_status(path, status):
    assert section_status(path) == status


def test_chunks_from_rejected_subsections_are_tagged():
    by_section = {c.section: c.status for c in chunk_documents([REJECTED_DOC], chunk_size=300, chunk_overlap=0)}
    assert by_section == {"Specification": "normative", "Using ``as`` in conditions": "rejected",
                          "Deferred Reference Counting": "normative"}
