from langchain_core.embeddings.fake import (
    FakeEmbeddings,
)

from langchain_chroma.vectorstores import Chroma


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    texts = ["foo", "bar", "baz"]
    Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
    )


def test_similarity_search() -> None:
    """Test similarity search by Chroma."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = Chroma.from_texts(
        collection_name="test_collection",
        texts=texts,
        embedding=FakeEmbeddings(size=10),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    docsearch.delete_collection()
    assert len(output) == 1


def test_get_by_ids_document_without_metadata() -> None:
    """Test that a document stored without metadata comes back with `{}`."""
    docsearch = Chroma.from_texts(
        collection_name="test_get_by_ids_without_metadata",
        texts=["foo", "bar"],
        embedding=FakeEmbeddings(size=10),
        metadatas=[{"page": "0"}, {}],
        ids=["1", "2"],
    )
    output = sorted(docsearch.get_by_ids(["1", "2"]), key=lambda doc: str(doc.id))
    docsearch.delete_collection()
    assert [(doc.id, doc.page_content, doc.metadata) for doc in output] == [
        ("1", "foo", {"page": "0"}),
        ("2", "bar", {}),
    ]
