import asyncio
import io
import threading
from functools import partial
from typing import cast

import flet as ft
import numpy as np
from matplotlib.backend_bases import Event, MouseButton, MouseEvent, PickEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from components.confirm_delete import confirm_delete
from utils.format_duration import format_duration
from utils.format_timestamp import format_timestamp
from utils.RecordingsDB import QuerySession, Recording, recordings_db


@ft.component
def RecordingDetailsPage():
    page = ft.context.page
    recording_id = ft.use_route_params().get("id", "")
    details, set_details = ft.use_state(cast(_Details | None, None))
    error, set_error = ft.use_state(cast(str | None, None))
    smoothed, set_smoothed = ft.use_state(cast(frozenset[int], frozenset()))
    images, set_images = ft.use_state(cast(dict[int, bytes], {}))
    chart_sizes = ft.use_memo(lambda: cast(dict[int, tuple[float, float]], {}), dependencies=[])

    async def load():
        try:
            result = await asyncio.to_thread(_load_details, int(recording_id))
        except Exception as e:
            set_error(f"Could not load the recording: {e}")
            return
        if result is None:
            set_error("This recording no longer exists.")
        else:
            details, images = result
            set_details(details)
            set_images(images)

    ft.use_effect(load, dependencies=[recording_id])

    async def toggle_smoothed(i: int, chart: "_SessionChart", e: ft.Event[ft.Switch]):
        value = bool(e.control.value)
        set_smoothed(lambda s: s | {i} if value else s - {i})
        image = await asyncio.to_thread(chart.set_smoothed, value)
        set_images(lambda images: {**images, i: image})

    async def tap_chart(i: int, chart: "_SessionChart", e: ft.TapEvent):
        size = chart_sizes.get(i)
        if size is None or e.local_position is None:
            return
        width, height = size
        image = await asyncio.to_thread(chart.tap, e.local_position.x / width, e.local_position.y / height)
        set_images(lambda images: {**images, i: image})

    def resize_chart(i: int, e: ft.LayoutSizeChangeEvent):
        chart_sizes[i] = (e.width, e.height)

    body: list[ft.Control]
    if error is not None:
        body = [ft.Text(error, color=ft.Colors.ERROR)]
    elif details is None:
        body = [ft.Container(alignment=ft.Alignment.CENTER, content=ft.ProgressRing())]
    else:
        recording, sessions = details
        body = [
            _labeled_field("Started", format_timestamp(recording.started_at)),
            _labeled_field("Ended", format_timestamp(recording.ended_at)),
            _labeled_field("Duration", format_duration(recording.ended_at - recording.started_at)),
        ]
        if not sessions:
            body.append(
                ft.Text(
                    "No keywords were searched for during this recording.",
                    size=18.0,
                    italic=True,
                )
            )
        for i, (session, chart) in enumerate(sessions):
            chart_controls: list[ft.Control] = []
            if session.scores:
                chart_controls.append(
                    ft.Switch(
                        # Switch has no label spacing option, so the label is padded instead.
                        label=ft.Container(padding=ft.Padding.only(left=12.0), content=ft.Text("Show moving average")),
                        value=i in smoothed,
                        on_change=partial(toggle_smoothed, i, chart),
                    )
                )
            chart_controls.append(
                ft.GestureDetector(
                    on_tap_down=partial(tap_chart, i, chart) if session.scores else None,
                    content=ft.Image(
                        src=images[i],
                        fit=ft.BoxFit.CONTAIN,
                        aspect_ratio=1.0,
                        gapless_playback=True,
                        on_size_change=partial(resize_chart, i),
                    ),
                )
            )
            if session.scores:
                chart_controls.append(
                    ft.Text(
                        "Tap a point to inspect it.",
                        size=14.0,
                        italic=True,
                        color=ft.Colors.ON_SURFACE_VARIANT,
                    )
                )
            session_controls = [_labeled_field("Keyword", session.prompt)]
            if session.baseline_prompt is not None:
                session_controls.append(_labeled_field("Baseline", session.baseline_prompt))
            session_controls.append(
                ft.Container(
                    padding=12.0,
                    border=ft.Border.all(1.0, ft.Colors.OUTLINE),
                    content=ft.Column(spacing=12.0, controls=chart_controls),
                )
            )
            body.append(ft.Column(key=str(i), spacing=12.0, controls=session_controls))

    return ft.View(
        route=f"/recordings/{recording_id}",
        appbar=ft.AppBar(
            title=ft.Text("Recording"),
            actions=[
                ft.IconButton(
                    icon=ft.Icons.DELETE_OUTLINE,
                    tooltip="Delete recording",
                    disabled=details is None,
                    on_click=lambda: confirm_delete(
                        int(recording_id),
                        lambda: page.navigate("/recordings"),
                        set_error,
                        None,
                    ),
                )
            ],
        ),
        controls=[
            ft.SafeArea(
                expand=True,
                content=ft.ListView(
                    expand=True,
                    padding=ft.Padding.symmetric(horizontal=16.0, vertical=12.0),
                    spacing=16.0,
                    controls=body,
                ),
            )
        ],
    )


_CHART_SIZE = 4.0
_CHART_DPI = 250.0
_LABEL_SIZE = 18.0
_TICK_SIZE = 16.0
_MOVING_AVERAGE_SAMPLES = 3
_PICK_RADIUS = 20.0


def _moving_average(session: QuerySession) -> list[float]:
    similarities = [s for _, s in session.scores]
    before, after = _MOVING_AVERAGE_SAMPLES // 2, (_MOVING_AVERAGE_SAMPLES - 1) // 2
    windows = (similarities[max(i - before, 0) : i + after + 1] for i in range(len(similarities)))
    return [sum(window) / len(window) for window in windows]


class _SessionChart:
    def __init__(self, recording: Recording, session: QuerySession):
        self._lock = threading.Lock()
        self._smoothed = False
        self._selected: int | None = None
        self._lines: list[Line2D] = []
        self._points: list[np.ndarray] = []
        self._fig = Figure(figsize=(_CHART_SIZE, _CHART_SIZE), dpi=_CHART_DPI, layout="constrained")
        ax = self._fig.subplots()
        if not session.scores:
            ax.text(
                0.5,
                0.75,
                "No measurements",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=_LABEL_SIZE,
            )
        else:
            times = [t - recording.started_at for t, _ in session.scores]
            measured = [s for _, s in session.scores]
            for similarities, label in (
                (measured, "Measured"),
                (_moving_average(session), "Moving Average"),
            ):
                (line,) = ax.plot(
                    times,
                    similarities,
                    color="C0",
                    linewidth=2.0,
                    marker="o" if len(session.scores) < 30 else None,
                    label=label,
                    picker=True,
                    pickradius=_PICK_RADIUS,
                )
                self._lines.append(line)
                self._points.append(np.column_stack([times, similarities]))
            self._crosshair = ax.axvline(0.0, color="C1", linewidth=1.5, linestyle="--")
            (self._highlight,) = ax.plot([], [], "o", color="C1", markersize=12.0, alpha=0.6)
            self._readout = ax.text(
                0.04,
                0.96,
                "",
                transform=ax.transAxes,
                va="top",
                fontsize=_TICK_SIZE,
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "C1", "alpha": 0.9},
            )
            # The y-axis is fitted to the measurements so it doesn't change when the moving average is toggled.
            low = min(measured)
            high = max(measured)
            margin = max((high - low) * 0.1, 0.05)
            ax.set_ylim(low - margin, high + margin)
            self._fig.canvas.mpl_connect("pick_event", self._on_pick)
        ended_at = session.ended_at or recording.ended_at
        ax.set_xlim(
            session.started_at - recording.started_at,
            max(ended_at, session.started_at + 1) - recording.started_at,
        )
        ax.set_xlabel("Seconds", fontsize=_LABEL_SIZE)
        ax.set_ylabel("Similarity", fontsize=_LABEL_SIZE)
        ax.tick_params(labelsize=_TICK_SIZE)
        # A handful of ticks keeps the large labels from colliding.
        ax.locator_params(nbins=5)
        ax.grid(alpha=0.3)

    def render(self) -> bytes:
        with self._lock:
            return self._render()

    def set_smoothed(self, smoothed: bool) -> bytes:
        with self._lock:
            self._smoothed = smoothed
            return self._render()

    def tap(self, x: float, y: float) -> bytes:
        with self._lock:
            self._selected = None
            if self._lines:
                event = MouseEvent(
                    "button_press_event",
                    self._fig.canvas,
                    round(x * self._fig.bbox.width),
                    round((1.0 - y) * self._fig.bbox.height),
                    button=MouseButton.LEFT,
                )
                self._fig.canvas.callbacks.process("button_press_event", event)
            return self._render()

    def _on_pick(self, event: Event) -> None:
        event = cast(PickEvent, event)
        indices: np.ndarray = getattr(event, "ind", np.array([], dtype=int))
        line = self._lines[self._smoothed]
        if event.artist is not line or not len(indices):
            return
        on_screen = self._fig.axes[0].transData.transform(self._points[self._smoothed][indices])
        distances = np.hypot(on_screen[:, 0] - event.mouseevent.x, on_screen[:, 1] - event.mouseevent.y)
        self._selected = int(indices[distances.argmin()])

    def _render(self) -> bytes:
        if self._lines:
            line = self._lines[self._smoothed]
            for other in self._lines:
                other.set_visible(other is line)
            for legend in list(self._fig.legends):
                legend.remove()
            self._fig.legend(handles=[line], loc="outside upper center", ncols=1, fontsize=_TICK_SIZE, frameon=False)
            selected = self._selected is not None
            self._crosshair.set_visible(selected)
            self._highlight.set_visible(selected)
            self._readout.set_visible(selected)
            if self._selected is not None:
                time, similarity = (float(v) for v in self._points[self._smoothed][self._selected])
                self._crosshair.set_xdata([time])
                self._highlight.set_data([time], [similarity])
                self._readout.set_text(f"t = {time:.1f} s\nsimilarity = {similarity:.3f}")
        buffer = io.BytesIO()
        self._fig.savefig(buffer, format="png")
        return buffer.getvalue()


# A recording and its query sessions, each paired with its interactive chart.
_Details = tuple[Recording, list[tuple[QuerySession, _SessionChart]]]


def _load_details(recording_id: int) -> tuple[_Details, dict[int, bytes]] | None:
    result = recordings_db.get_recording(recording_id)
    if result is None:
        return None
    recording, sessions = result
    charts = [_SessionChart(recording, session) for session in sessions]
    images = {i: chart.render() for i, chart in enumerate(charts)}
    return (recording, list(zip(sessions, charts))), images


def _labeled_field(label: str, value: str) -> ft.Control:
    return ft.Column(
        spacing=2.0,
        controls=[
            ft.Text(label, size=16.0, weight=ft.FontWeight.BOLD),
            ft.Text(value, size=20.0),
        ],
    )
