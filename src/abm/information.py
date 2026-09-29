"""Causal message delivery on directed information edges."""

from __future__ import annotations

import heapq
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np

from .schemas import Edge, JsonValue, Message, SchemaMixin


@dataclass(frozen=True, slots=True)
class Delivery(SchemaMixin):
    message_id: str
    source_id: str
    target_id: str
    event_time: datetime
    available_at: datetime
    content_type: str
    payload: dict[str, JsonValue]
    credibility: float
    path: tuple[str, ...]


class InformationGraph:
    def __init__(self, nodes: tuple[str, ...], edges: tuple[Edge, ...]):
        if len(set(nodes)) != len(nodes):
            raise ValueError("graph nodes must be unique")
        if any(edge.source_id not in nodes or edge.target_id not in nodes for edge in edges):
            raise ValueError("edge endpoints must be graph nodes")
        self.nodes = nodes
        self.edges = edges
        self.current_time: datetime | None = None
        self._messages: dict[str, Message] = {}
        self._queue: list[tuple] = []
        self._seen: set[tuple[str, str]] = set()
        self._deliveries: list[Delivery] = []
        self._sequence = 0
        self._index_edges()

    def _index_edges(self) -> None:
        self._outgoing = {node: [] for node in self.nodes}
        for edge in sorted(self.edges, key=lambda e: (e.source_id, e.target_id, e.valid_from, e.edge_type)):
            self._outgoing[edge.source_id].append(edge)

    def publish(self, message: Message) -> None:
        if message.message_id in self._messages:
            raise ValueError("message has already been published")
        if message.source_id not in self.nodes:
            raise ValueError("message source must be a graph node")
        if any(target != "public" and target not in self.nodes for target in message.target_scope):
            raise ValueError("message scope contains an unknown node")
        if self.current_time is not None and message.available_at < self.current_time:
            raise ValueError("messages cannot be published into the past")
        message = replace(message, payload=deepcopy(message.payload))
        self._messages[message.message_id] = message
        self._seen.add((message.message_id, message.source_id))
        origin = Delivery(
            message.message_id, message.source_id, message.source_id,
            message.event_time, message.available_at, message.content_type,
            deepcopy(message.payload), message.credibility, (message.source_id,),
        )
        self._sequence += 1
        heapq.heappush(self._queue, (
            origin.available_at, -origin.credibility, origin.path,
            origin.message_id, self._sequence, origin,
        ))

    def _forward(self, message: Message, at: datetime, credibility: float, path: tuple[str, ...]) -> None:
        for edge in self._outgoing[path[-1]]:
            if edge.target_id in path:
                continue
            if "public" not in message.target_scope and edge.target_id not in message.target_scope:
                continue
            if at.date() < edge.valid_from or (edge.valid_to is not None and at.date() > edge.valid_to):
                continue
            trust = credibility * edge.trust * edge.weight
            if trust <= 0:
                continue
            delivery = Delivery(
                message.message_id, message.source_id, edge.target_id,
                message.event_time, at + timedelta(days=edge.delay_days),
                message.content_type, deepcopy(message.payload), trust,
                path + (edge.target_id,),
            )
            self._sequence += 1
            heapq.heappush(self._queue, (
                delivery.available_at, -trust, delivery.path,
                message.message_id, self._sequence, delivery,
            ))

    def advance(self, now: datetime) -> tuple[Delivery, ...]:
        if self.current_time is not None and now < self.current_time:
            raise ValueError("graph clock cannot move backwards")
        self.current_time = now
        arrived = []
        while self._queue and self._queue[0][0] <= now:
            *_, delivery = heapq.heappop(self._queue)
            if len(delivery.path) == 1:
                self._forward(self._messages[delivery.message_id], delivery.available_at,
                              delivery.credibility, delivery.path)
                continue
            key = (delivery.message_id, delivery.target_id)
            if key in self._seen:
                continue
            self._seen.add(key)
            self._deliveries.append(delivery)
            arrived.append(replace(delivery, payload=deepcopy(delivery.payload)))
            self._forward(
                self._messages[delivery.message_id], delivery.available_at,
                delivery.credibility, delivery.path,
            )
        return tuple(arrived)

    def visible(self, node: str, now: datetime) -> tuple[Delivery, ...]:
        if node not in self.nodes:
            raise ValueError("unknown graph node")
        if self.current_time is None or now > self.current_time:
            raise ValueError("visibility cannot advance the graph clock")
        return tuple(
            replace(delivery, payload=deepcopy(delivery.payload))
            for delivery in self._deliveries
            if delivery.target_id == node and delivery.available_at <= now
        )

    def active_edges(self, now: datetime) -> tuple[Edge, ...]:
        return tuple(edge for edge in self.edges if edge.valid_from <= now.date()
                     and (edge.valid_to is None or now.date() <= edge.valid_to))

    def adapt_from_trades(
        self, trader_ids: tuple[str, ...], trades: np.ndarray,
        observed_at: tuple[datetime, ...], effective_at: datetime, rate: float = 0.1,
    ) -> tuple[dict[str, JsonValue], ...]:
        if self.current_time is None or effective_at.date() <= self.current_time.date():
            raise ValueError("dynamic edges must take effect after the current day")
        if not observed_at or any(at > self.current_time for at in observed_at):
            raise ValueError("dynamic edges can only use observed trades")
        if trades.shape != (len(observed_at), len(trader_ids)) or not np.all(np.isfinite(trades)):
            raise ValueError("trade history must match its timestamps and traders")
        if not 0 <= rate <= 1:
            raise ValueError("edge adaptation rate must be between zero and one")
        indices = {node: index for index, node in enumerate(trader_ids)}
        edges = []
        updates = []
        active = set(self.active_edges(self.current_time))
        for edge in self.edges:
            if (edge not in active or edge.source_id not in indices or edge.target_id not in indices
                    or (edge.valid_to is not None and edge.valid_to < effective_at.date())):
                edges.append(edge)
                continue
            left, right = trades[:, indices[edge.source_id]], trades[:, indices[edge.target_id]]
            norm = float(np.linalg.norm(left) * np.linalg.norm(right))
            similarity = (0.5 + 0.5 * float(np.clip(np.dot(left, right) / norm, -1, 1))) if norm > 0 else edge.weight
            weight = (1.0 - rate) * edge.weight + rate * similarity if norm > 0 else edge.weight
            edges.append(replace(edge, valid_to=effective_at.date() - timedelta(days=1)))
            edges.append(replace(edge, weight=weight, valid_from=effective_at.date()))
            updates.append({
                "source_id": edge.source_id, "target_id": edge.target_id,
                "before_weight": edge.weight, "after_weight": weight,
                "observed_through": max(observed_at).isoformat(),
                "effective_at": effective_at.isoformat(),
            })
        self.edges = tuple(edges)
        self._index_edges()
        return tuple(updates)
