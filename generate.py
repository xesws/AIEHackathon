"""Inference entrypoint: build the prompt from memory state, then decode on the EDITED model.

    generate(query, *, model, buffer, rag_hits, with_rag=True) -> str

Greedy decoding mirrors HoReN's eval (``test_prediction_acc``): ``do_sample=False`` +
``use_cache=False``. When ``model`` is a HoReN wrapper, ``model.generate`` first sets the
adapter ``key_id`` to the last prompt token so retrieval is positioned correctly.

``use_chat_template`` toggles the prompt format:
  - ``False`` -> raw prompt (the exact format the HoReN edit was trained on).
  - ``True``  -> rendered via ``memory.prompt.build_prompt`` (Engram's inference skeleton).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from memory.prompt import build_prompt
from memory.schema import MemoryItem


def _greedy(model: Any, tok: Any, prompt_text: str, *, max_new_tokens: int, device: str) -> str:
    import torch

    enc = tok(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            use_cache=False,
        )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).lstrip()


def generate(
    query: str,
    *,
    model: Any,
    buffer: Sequence[MemoryItem] = (),
    rag_hits: Sequence[MemoryItem] = (),
    with_rag: bool = True,
    tok: Optional[Any] = None,
    max_new_tokens: int = 16,
    use_chat_template: bool = False,
    device: str = "cuda:0",
) -> str:
    """Build the prompt and greedily decode from ``model``. See module docstring for modes."""
    if tok is None:
        import serving.model_host as model_host

        tok = model_host.tokenizer()

    if use_chat_template:
        messages = build_prompt(
            query,
            buffer if with_rag else [],
            rag_hits if with_rag else [],
        )
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = query

    return _greedy(model, tok, prompt_text, max_new_tokens=max_new_tokens, device=device)
