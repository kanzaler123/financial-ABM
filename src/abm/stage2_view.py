"""Read-only native views over persisted Stage 2 daily frames."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets
from vispy import scene

from .tabular import COMPANY_ACTIONS, TRADER_ACTIONS


class VisPyNetworkView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.caption = QtWidgets.QLabel("Information graph · arrows follow source to target")
        layout.addWidget(self.caption)
        self.canvas = scene.SceneCanvas(keys=None, bgcolor="white", show=False)
        view = self.canvas.central_widget.add_view()
        view.camera = scene.PanZoomCamera(rect=(-1.4, -1.3, 2.8, 2.6))
        self.lines = scene.visuals.Arrow(parent=view.scene, arrow_size=8, connect="segments")
        self.paths = scene.visuals.Arrow(parent=view.scene, arrow_size=10, connect="segments", color="#D97706", arrow_color="#D97706", width=3)
        self.nodes = scene.visuals.Markers(parent=view.scene)
        self.labels = scene.visuals.Text(parent=view.scene, color="#1F2937", font_size=9)
        layout.addWidget(self.canvas.native, 1)
        self.visible_nodes = []

    def set_frame(self, frame, trader, company, source):
        nodes = frame["nodes"]
        if len(nodes) > 60:
            selected = {trader, company}
            neighbors = {e["target_id"] for e in frame["edges"] if e["source_id"] == trader}
            neighbors.update(e["source_id"] for e in frame["edges"] if e["target_id"] == trader)
            nodes = sorted(selected | neighbors)
        self.visible_nodes = nodes
        self.caption.setText(f"Information graph · {len(nodes)} / {len(frame['nodes'])} nodes · orange = arrivals today")
        if not nodes:
            self.lines.set_data(pos=np.empty((0, 2)), arrows=np.empty((0, 4)))
            self.paths.set_data(pos=np.empty((0, 2)), arrows=np.empty((0, 4)))
            self.nodes.set_data(pos=np.empty((0, 2)))
            self.labels.text = "Graph disabled: direct public broadcast"
            self.labels.pos = (0, 0)
            return
        angles = np.arange(len(nodes)) * 2 * np.pi / len(nodes)
        positions = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        index = {node: i for i, node in enumerate(nodes)}
        segments = [(positions[index[e['source_id']]], positions[index[e['target_id']]])
                    for e in frame['edges'] if e['source_id'] in index and e['target_id'] in index]
        edges = np.asarray(segments, dtype=np.float32).reshape(-1, 2)
        self.lines.set_data(pos=edges, arrows=edges.reshape(-1, 4), color="#A8B4C2")
        paths = []
        for delivery in frame['arrivals']:
            if source != 'all' and delivery['source_id'] != source:
                continue
            for start, end in zip(delivery['path'], delivery['path'][1:]):
                if start in index and end in index:
                    paths.extend((positions[index[start]], positions[index[end]]))
        paths = np.asarray(paths, dtype=np.float32).reshape(-1, 2)
        self.paths.set_data(pos=paths, arrows=paths.reshape(-1, 4))
        colors = np.array([(0.79, 0.58, 0.10, 1) if n.startswith('company') else (0.18, 0.37, 0.55, 1) for n in nodes])
        self.nodes.set_data(pos=positions, face_color=colors, edge_color='white', size=16)
        self.labels.text = nodes
        self.labels.pos = positions * 1.16
        self.canvas.update()


class Stage2Panel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames = {}
        self.current_frame = None
        layout = QtWidgets.QVBoxLayout(self)
        filters = QtWidgets.QHBoxLayout()
        self.trader = QtWidgets.QComboBox()
        self.company = QtWidgets.QComboBox()
        self.source = QtWidgets.QComboBox()
        self.start_day = QtWidgets.QSpinBox()
        self.end_day = QtWidgets.QSpinBox()
        self.day = QtWidgets.QSpinBox()
        self.follow = QtWidgets.QCheckBox("Follow latest")
        self.follow.setChecked(True)
        for label, widget in (("Trader", self.trader), ("Company", self.company), ("Source", self.source),
                              ("From", self.start_day), ("To", self.end_day), ("Day", self.day)):
            filters.addWidget(QtWidgets.QLabel(label))
            filters.addWidget(widget)
            if isinstance(widget, QtWidgets.QSpinBox):
                widget.setRange(1, 100000)
                widget.valueChanged.connect(self.refresh)
            else:
                widget.currentTextChanged.connect(self.refresh)
        filters.addWidget(self.follow)
        layout.addLayout(filters)
        self.network = VisPyNetworkView()
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        top = QtWidgets.QSplitter()
        top.addWidget(self.network)
        top.addWidget(self.details)
        top.setSizes([650, 450])
        layout.addWidget(top, 3)
        grid = QtWidgets.QGridLayout()
        self.plots = {}
        for i, title in enumerate(("Action shares", "Wealth distribution", "Bandit choice probabilities",
                                    "Company Q values", "Trader reward components", "Company reward components")):
            plot = pg.PlotWidget(title=title)
            plot.showGrid(x=False, y=True, alpha=0.2)
            self.plots[title] = plot
            grid.addWidget(plot, i // 3, i % 3)
        layout.addLayout(grid, 4)

    def set_events(self, events):
        incoming = [e for e in events if e.sequence_no not in self.frames and "stage2_json" in e.payload]
        if not incoming:
            return
        for event in incoming:
            self.frames[event.sequence_no] = json.loads(event.payload["stage2_json"])
        latest = self.frames[max(self.frames)]
        if not self.trader.count():
            for combo, values in ((self.trader, latest['trader_ids']), (self.company, latest['company_ids']),
                                  (self.source, ['all'] + latest['company_ids'])):
                combo.blockSignals(True)
                combo.addItems(values)
                combo.blockSignals(False)
        if self.follow.isChecked():
            for widget in (self.end_day, self.day):
                widget.blockSignals(True)
                widget.setValue(latest['day'])
                widget.blockSignals(False)
        self.refresh()

    def _bars(self, title, labels, values):
        plot = self.plots[title]
        plot.clear()
        plot.getAxis('bottom').setTicks([list(enumerate(labels))])
        plot.addItem(pg.BarGraphItem(x=np.arange(len(values)), height=values, width=0.65, brush="#2F5D8C"))

    def refresh(self):
        if not self.frames or not self.trader.count():
            return
        frames = [f for f in self.frames.values() if self.start_day.value() <= f['day'] <= self.end_day.value()]
        candidates = [f for f in frames if f['day'] <= self.day.value()]
        if not candidates:
            self.current_frame = None
            self.details.setPlainText("No recorded frame in this time range.")
            return
        frame = candidates[-1]
        self.current_frame = frame
        i, j = self.trader.currentIndex(), self.company.currentIndex()
        self.network.set_frame(frame, self.trader.currentText(), self.company.currentText(), self.source.currentText())
        counts, bins = np.histogram(frame['wealth'], bins=min(20, len(frame['wealth'])))
        plot = self.plots['Wealth distribution']
        plot.clear()
        plot.getAxis('bottom').setTicks(None)
        plot.addItem(pg.BarGraphItem(x=(bins[1:] + bins[:-1]) / 2, height=counts, width=np.diff(bins) * .9, brush="#C9941A"))
        actions = frame['trader_actions']
        self._bars('Action shares', TRADER_ACTIONS, [actions.count(a) / len(actions) if actions else 0 for a in range(4)])
        probabilities = frame['trader_probabilities'][i] if frame['trader_probabilities'] else [0] * 4
        self._bars('Bandit choice probabilities', TRADER_ACTIONS, probabilities)
        q = frame['q_values'][frame['company_groups'][j]][frame['company_states'][j]]
        self._bars('Company Q values', COMPANY_ACTIONS, q)
        trader_rewards = [f['trader_rewards'] for f in candidates if f['trader_rewards']]
        company_rewards = [f['company_rewards'] for f in candidates if f['company_rewards']]
        tr = trader_rewards[-1] if trader_rewards else None
        cr = company_rewards[-1] if company_rewards else None
        self._bars('Trader reward components', ('gross', 'fee', 'volatility', 'drawdown'),
                   [tr[k][i] for k in ('gross_return', 'transaction_cost', 'volatility_penalty', 'drawdown_penalty')] if tr else [0] * 4)
        self._bars('Company reward components', ('value', 'finance', 'real', 'disclosure', 'penalty'), cr['components'][j] if cr else [0] * 5)
        detail = {'day': frame['day'], 'trader': self.trader.currentText(),
                  'decision_state': frame['trader_states'][i], 'observed_features_before_trade': frame['trader_features'][i],
                  'arrived_news': frame['visible_news'][i], 'subjective_value_at_close': frame['subjective_values'][i],
                  'action': TRADER_ACTIONS[actions[i]] if actions else 'fixed Stage 1 rule',
                  'order': frame['submitted_orders'][i], 'executed': frame['executed_orders'][i],
                  'position': frame['positions'][i], 'wealth': frame['wealth'][i],
                  'bandit_values': frame['bandit_values'][frame['trader_types'][i]][frame['trader_states'][i]],
                  'company': self.company.currentText(), 'company_action': COMPANY_ACTIONS[frame['company_actions'][j]],
                  'company_state': frame['company_states'][j], 'company_cash': frame['company_cash'][j],
                  'company_private_features_research_only': frame['company_features'][j],
                  'trader_reward_window': [tr['start_day'], tr['end_day']] if tr else None,
                  'company_reward_window': [cr['start_day'], cr['end_day']] if cr else None}
        deliveries = [d for f in candidates for d in f['arrivals'] if d['target_id'] == self.trader.currentText()
                      and (self.source.currentText() == 'all' or d['source_id'] == self.source.currentText())]
        detail['arrivals_in_range'] = [{**d, 'delay_days': (datetime.fromisoformat(d['available_at']) - datetime.fromisoformat(d['event_time'])).days} for d in deliveries]
        self.details.setPlainText(json.dumps(detail, indent=2, ensure_ascii=False))
