"""Engram continual-learning memory system (core IP).

Two-speed memory: a small short-term ``buffer`` (route==edit, later consolidated into weights)
plus a permanent ``rag_store`` (route==rag). ``extract`` pulls candidates from chat, ``router``
routes each, ``dedup`` + ``consolidate`` fold buffer items into the model's weights via
``editing.edit``, and ``prompt`` assembles the fixed inference skeleton.
"""
