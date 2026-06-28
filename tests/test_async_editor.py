from __future__ import annotations

import time


def _wait_for_job(async_editor, job_id: str, status: str = "succeeded") -> dict:
    deadline = time.time() + 2.0
    while time.time() < deadline:
        job = async_editor.get(job_id)
        if job and job["status"] == status:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {status}: {async_editor.get(job_id)!r}")


def test_async_editor_runs_consolidate_with_shadow_backend(monkeypatch):
    from serving import async_editor, shadow_editing

    async_editor.stop()
    with async_editor._lock:
        async_editor._jobs.clear()

    calls = []

    def fake_run_pass(trigger, ids=None, editing_module=None):
        calls.append({"trigger": trigger, "ids": ids, "editing_module": editing_module})
        return 3

    monkeypatch.setattr(async_editor.consolidate, "run_pass", fake_run_pass)

    queued = async_editor.submit(trigger="manual", ids=["m1"])
    done = _wait_for_job(async_editor, queued["id"])

    assert done["status"] == "succeeded"
    assert done["n_written"] == 3
    assert calls == [{"trigger": "manual", "ids": ["m1"], "editing_module": shadow_editing}]

    async_editor.stop()


def test_async_editor_records_failures(monkeypatch):
    from serving import async_editor

    async_editor.stop()
    with async_editor._lock:
        async_editor._jobs.clear()

    def fake_run_pass(trigger, ids=None, editing_module=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(async_editor.consolidate, "run_pass", fake_run_pass)

    queued = async_editor.submit(trigger="manual")
    failed = _wait_for_job(async_editor, queued["id"], status="failed")

    assert failed["n_written"] is None
    assert "RuntimeError" in failed["error"]
    assert "boom" in failed["error"]

    async_editor.stop()
