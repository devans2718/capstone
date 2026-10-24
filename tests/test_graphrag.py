"""kg/graphrag.py's pipeline, with the model and the graph stubbed.

The app's Ask the graph tab calls route -> retrieve -> answer unchanged, so
this checks the three of them hand over to each other as the tab relies on:
the route's JSON parsed into a choice, the choice's parameters reaching every
statement of that retrieval, the rows keyed by statement name, and those rows
in the answer's prompt. No network and no Neo4j.

App environment only: neo4j and anthropic come from requirements-app.txt.
kg/ is not a package, so the module is loaded from its path, as the app does.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("neo4j", reason="app environment only (requirements-app.txt)")
pytest.importorskip("anthropic", reason="app environment only (requirements-app.txt)")

GRAPHRAG = Path(__file__).resolve().parents[1] / "kg" / "graphrag.py"


class Record:
    def __init__(self, data: dict):
        self._data = data

    def data(self) -> dict:
        return self._data


class Driver:
    """Stands in for GraphDatabase.driver(...): records each query it is sent."""

    def __init__(self, sent: list):
        self.sent = sent

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_query(self, cypher: str, params: dict):
        self.sent.append((cypher, params))
        return [Record({"key": "crit-2-r3", "n": len(self.sent)})], None, None


def test_route_retrieve_answer_hand_over(monkeypatch):
    spec = importlib.util.spec_from_file_location("graphrag", GRAPHRAG)
    graphrag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graphrag)

    calls = []

    def complete(system, user, schema=None, effort="medium"):
        calls.append({"system": system, "user": user, "schema": schema})
        if schema is not None:
            return json.dumps({"retrieval": "why_failed", "term": "",
                               "criterion": 0})
        return "Criterion 2 failed [crit-2-r3]."

    sent = []
    monkeypatch.setattr(graphrag, "complete", complete)
    monkeypatch.setattr(graphrag.GraphDatabase, "driver",
                        lambda uri, auth: Driver(sent))

    question = "Why did the pilot fail?"
    choice = graphrag.route(question)
    rows = graphrag.retrieve(choice)
    text = graphrag.answer(question, rows)

    assert choice == {"retrieval": "why_failed", "term": "", "criterion": 0}
    statements = graphrag.RETRIEVALS["why_failed"]["statements"]
    assert [c for c, _ in sent] == list(statements.values())
    assert all(p == {"term": "", "criterion": 0} for _, p in sent)
    assert list(rows) == list(statements)
    assert rows["failed_criteria"] == [{"key": "crit-2-r3", "n": 1}]

    route_call, answer_call = calls
    assert route_call["schema"]["properties"]["retrieval"]["enum"] == list(graphrag.RETRIEVALS)
    assert question in answer_call["user"] and '"crit-2-r3"' in answer_call["user"]
    assert answer_call["schema"] is None
    assert text == "Criterion 2 failed [crit-2-r3]."
