"""Local persistence coverage for RAG storage and HoReN codebook checkpoints."""
from __future__ import annotations

import torch

from memory import embed, rag_store, store
from memory.schema import MemoryItem
from serving import model_host


def _item(
    item_id: str,
    text: str,
    *,
    route: str = "rag",
    status: str = "consolidated",
) -> MemoryItem:
    return MemoryItem(
        id=item_id,
        type="fact",
        text=text,
        route=route,
        status=status,
        source="test",
        ts=1.0,
        provenance=None,
    )


def test_store_persists_edit_and_rag_records(tmp_path):
    store.reset()
    store.enable_persistence(data_dir=tmp_path)

    edit_item = _item("e1", "JQ likes concise answers.", route="edit", status="buffer")
    rag_item = _item("r1", "The spec says scheduling uses CP-SAT.")
    store.upsert(edit_item)
    store.rag_add(rag_item, [0.1, 0.2, 0.3])

    assert (tmp_path / "memory_store.json").exists()

    store.reset()
    store.enable_persistence(data_dir=tmp_path)

    assert store.get("e1").text == edit_item.text
    restored_rag = store.rag_all()
    assert [(it.id, vec) for it, vec in restored_rag] == [("r1", [0.1, 0.2, 0.3])]
    store.reset()


def test_rag_chunks_persist_and_restore_search(tmp_path, monkeypatch):
    store.reset()
    rag_store.reset()

    table = {
        "The spec says scheduling uses CP-SAT.": [1.0, 0.0],
        "Which solver handles scheduling?": [1.0, 0.0],
    }

    def fake_encode(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [list(table[t]) for t in texts]

    monkeypatch.setattr(embed, "encode", fake_encode)
    store.enable_persistence(data_dir=tmp_path)
    rag_store.enable_persistence(data_dir=tmp_path)

    rag_store.add(_item("r1", "The spec says scheduling uses CP-SAT."))
    assert (tmp_path / "rag_chunks.json").exists()

    store.reset()
    rag_store.reset()
    store.enable_persistence(data_dir=tmp_path)
    rag_store.enable_persistence(data_dir=tmp_path)

    hits = rag_store.search("Which solver handles scheduling?", k=5)

    assert [h.id for h in hits] == ["r1"]
    store.reset()
    rag_store.reset()


class _FakeAdapter:
    def __init__(self):
        self.adapter_mode = "value"
        self.device = torch.device("cpu")
        self.keys = torch.zeros(1, 3, dtype=torch.float32)
        self.values = torch.nn.Parameter(torch.zeros(1, 4, dtype=torch.float32))
        self.lora_A = None
        self.lora_B = None
        self.key_labels = [torch.tensor(-1)]
        self.normalize_codebook_keys = True
        self.query_selection_strategy = "last_prompt_token"
        self.query_span_pool_strategy = "flat"
        self.hopfield_key_match_threshold = 0.85


def test_codebook_checkpoint_round_trips_adapter_state(tmp_path):
    adapter = _FakeAdapter()
    adapter.keys = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    adapter.values = torch.nn.Parameter(
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 3.0, 4.0, 5.0]])
    )
    adapter.key_labels = [torch.tensor(-1), torch.tensor([7, 8, 9])]

    path = tmp_path / "codebook.pt"
    assert model_host.save_codebook(adapter, path=path) is True

    restored = _FakeAdapter()
    state = torch.load(path, map_location="cpu")
    model_host._apply_codebook_state(restored, state)

    assert torch.equal(restored.keys, adapter.keys)
    assert torch.equal(restored.values.detach(), adapter.values.detach())
    assert len(restored.key_labels) == 2
    assert torch.equal(restored.key_labels[1], torch.tensor([7, 8, 9]))
