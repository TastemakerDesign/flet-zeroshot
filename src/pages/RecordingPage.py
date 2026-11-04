import asyncio
import time
from dataclasses import replace
from typing import cast

import flet as ft
import flet_camera as fc
import numpy as np

from components.record_button import record_button
from utils.crop_square import crop_square
from utils.encode_jpeg import encode_jpeg
from utils.FastSamDetector import Detection, FastSamDetector
from utils.is_camera_supported import is_camera_supported
from utils.RecordingsDB import QuerySession, recordings_db
from utils.SettingsManager import settings_manager

_DETECTION_THRESHOLD = 0.25
_MIN_THRESHOLD = 0.10
_MAX_THRESHOLD = 0.30
_THRESHOLD_SETTING = "threshold"
_VIBRATION_INTERVAL = 0.4
_FLASH_DURATION = 0.15
_BORDER_WIDTH = 4.0
_BORDER_COLOR = ft.Colors.GREY_800
_BOX_WIDTH = 4.0
_BOX_COLOR = ft.Colors.RED
_BOX_ANIMATION = ft.Animation(150, ft.AnimationCurve.EASE_OUT)
_MIN_STATUS_SPACE = 48.0

# The main query's embedding and the optional baseline query's embedding.
_TextEmbeddings = tuple[np.ndarray, np.ndarray | None]


@ft.component
def RecordingPage():
    page = ft.context.page
    supported = is_camera_supported(page)
    camera = ft.use_memo(lambda: ft.Ref[fc.Camera](), [])
    sensor_orientation = ft.use_ref(0)
    busy = ft.use_ref(False)
    detector = ft.use_ref(cast(FastSamDetector | None, None))
    text_embeddings = ft.use_ref(cast(_TextEmbeddings | None, None))
    detecting = ft.use_ref(False)
    detect_task = ft.use_ref(None)
    haptics = ft.use_ref(None)
    vibration_enabled = ft.use_ref(True)
    flash_task = ft.use_ref(None)
    last_vibration = ft.use_ref(0.0)
    latest_query = ft.use_ref("")
    latest_baseline = ft.use_ref("")
    latest_threshold = ft.use_ref(_DETECTION_THRESHOLD)
    started_at = ft.use_memo(time.time, [])
    finished_sessions = ft.use_memo(list[QuerySession], [])
    active_session = ft.use_ref(cast(QuerySession | None, None))
    saving, set_saving = ft.use_state(False)
    save_error, set_save_error = ft.use_state(cast(str | None, None))
    frame, set_frame = ft.use_state(cast(bytes | None, None))
    draft, set_draft = ft.use_state("")
    query, set_query = ft.use_state("")
    baseline_draft, set_baseline_draft = ft.use_state("")
    baseline, set_baseline = ft.use_state("")
    score, set_score = ft.use_state(cast(float | None, None))
    threshold, set_threshold = ft.use_state(_DETECTION_THRESHOLD)
    detection, set_detection = ft.use_state(cast(Detection | None, None))
    flashing, set_flashing = ft.use_state(False)
    area_size, set_area_size = ft.use_state((0.0, 0.0))
    search_generation, set_search_generation = ft.use_state(0)
    model_ready, set_model_ready = ft.use_state(False)
    show_inference_times, set_show_inference_times = ft.use_state(False)
    inference_times, set_inference_times = ft.use_state(cast(tuple[float | None, float | None], (None, None)))
    model_error, set_model_error = ft.use_state(cast(str | None, None))
    error, set_error = ft.use_state(None if supported else "The camera is only supported on Android and iOS.")
    latest_query.current = query
    latest_baseline.current = baseline
    latest_threshold.current = threshold

    def current_camera() -> fc.Camera:
        if camera.current is None:
            raise RuntimeError("The camera control isn't mounted.")
        return camera.current

    async def start_camera():
        if not supported:
            return
        try:
            cameras = await current_camera().get_available_cameras()
            if not cameras:
                set_error("No camera was found on this device.")
                return
            back = next(
                (c for c in cameras if c.lens_direction == fc.CameraLensDirection.BACK),
                cameras[0],
            )
            sensor_orientation.current = back.sensor_orientation - 90
            resolution = await asyncio.to_thread(settings_manager.get_resolution)
            await current_camera().initialize(back, resolution, enable_audio=False)
            if not await current_camera().supports_image_streaming():
                set_error("This device doesn't support streaming camera frames.")
                return
            await current_camera().start_image_stream()
        except Exception as e:
            set_error(f"Could not start the camera: {e}")

    async def stop_camera():
        if camera.current is not None:
            try:
                await camera.current.stop_image_stream()
            except Exception:
                pass

    ft.use_effect(start_camera, dependencies=[], cleanup=stop_camera)

    async def load_model():
        try:
            precision = await asyncio.to_thread(settings_manager.get_model_precision)
            detector.current = await asyncio.to_thread(FastSamDetector, precision)
            set_model_ready(True)
        except Exception as e:
            set_model_error(f"Could not load the model: {e}")

    ft.use_effect(load_model, dependencies=[])

    async def load_settings():
        try:
            vibration_enabled.current = await asyncio.to_thread(settings_manager.get_vibration)
            set_show_inference_times(await asyncio.to_thread(settings_manager.get_inference_overlay))
            set_threshold(await asyncio.to_thread(_load_threshold))
        except Exception:
            pass

    ft.use_effect(load_settings, dependencies=[])

    def switch_session(prompt: str | None, baseline_prompt: str | None):
        now = time.time()
        if active_session.current is not None:
            active_session.current.ended_at = now
            finished_sessions.append(active_session.current)
        active_session.current = QuerySession(prompt, baseline_prompt, now) if prompt else None

    async def update_text_embedding():
        prompt = query.strip()
        baseline_prompt = baseline.strip() or None
        if not prompt or detector.current is None:
            text_embeddings.current = None
            switch_session(None, None)
            set_score(None)
            set_detection(None)
            return
        embedding = await asyncio.to_thread(detector.current.encode_text, prompt)
        baseline_embedding = None
        if baseline_prompt is not None:
            baseline_embedding = await asyncio.to_thread(detector.current.encode_text, baseline_prompt)
        latest_baseline_prompt = (latest_baseline.current or "").strip() or None
        if prompt == (latest_query.current or "").strip() and baseline_prompt == latest_baseline_prompt:
            text_embeddings.current = (embedding, baseline_embedding)
            switch_session(prompt, baseline_prompt)
            set_score(None)
            set_detection(None)

    ft.use_effect(update_text_embedding, dependencies=[query, baseline, model_ready])

    async def stop_recording():
        set_saving(True)
        set_save_error(None)
        ended_at = time.time()
        sessions = list(finished_sessions)
        if active_session.current is not None:
            sessions.append(replace(active_session.current, ended_at=ended_at))
        try:
            await asyncio.to_thread(recordings_db.save_recording, started_at, ended_at, sessions)
        except Exception as e:
            set_saving(False)
            set_save_error(f"Could not save the recording: {e}")
            return
        ft.context.page.navigate("/recordings")

    async def flash():
        set_flashing(True)
        await asyncio.sleep(_FLASH_DURATION)
        set_flashing(False)

    async def signal_detection():
        now = time.monotonic()
        last = last_vibration.current
        if last is not None and now - last < _VIBRATION_INTERVAL:
            return
        last_vibration.current = now
        flash_task.current = asyncio.create_task(flash())
        if not vibration_enabled.current:
            return
        try:
            if haptics.current is None:
                haptics.current = ft.HapticFeedback()
            await haptics.current.vibrate()
        except Exception:
            pass

    async def detect(model: FastSamDetector, square, embeddings: _TextEmbeddings, captured_at):
        try:
            found = await asyncio.to_thread(model.detect, square, *embeddings)
        finally:
            detecting.current = False
        set_inference_times(model.inference_times())
        if text_embeddings.current is not embeddings:
            return
        if found is None:
            set_score(None)
            set_detection(None)
            return
        set_score(found.similarity)
        set_detection(found)
        if active_session.current is not None:
            active_session.current.scores.append((captured_at, found.similarity))
        if found.is_match(latest_threshold.current or _DETECTION_THRESHOLD):
            await signal_detection()

    async def on_stream_image(e: fc.CameraImageEvent):
        if busy.current:
            return
        busy.current = True
        captured_at = time.time()
        try:
            square, jpeg = await asyncio.to_thread(_prepare_frame, e.bytes, sensor_orientation.current or 0)
            if jpeg is not None:
                set_frame(jpeg)
            model = detector.current
            embeddings = text_embeddings.current
            if square is not None and model is not None and embeddings is not None and not detecting.current:
                detecting.current = True
                detect_task.current = asyncio.create_task(detect(model, square, embeddings, captured_at))
        finally:
            busy.current = False

    area_width, area_height = area_size
    side = max(min(area_width, area_height - 2.0 * _MIN_STATUS_SPACE), 0.0)

    preview: ft.Control
    if error is not None:
        preview = ft.Container(
            alignment=ft.Alignment.CENTER,
            padding=24.0,
            content=ft.Text(error, color=ft.Colors.WHITE, size=18.0, text_align=ft.TextAlign.CENTER),
        )
    elif frame is None:
        preview = ft.Container(
            alignment=ft.Alignment.CENTER,
            content=ft.ProgressRing(color=ft.Colors.WHITE),
        )
    else:
        inner = side - 2.0 * _BORDER_WIDTH
        overlay: list[ft.Control] = [
            ft.Image(src=frame, fit=ft.BoxFit.CONTAIN, gapless_playback=True),
        ]
        if score is not None and detection is not None and detection.is_match(threshold):
            color = ft.Colors.BLACK if flashing else _BOX_COLOR
            overlay.append(
                ft.Container(
                    key="box",
                    left=detection.left * inner,
                    top=detection.top * inner,
                    width=(detection.right - detection.left) * inner,
                    height=(detection.bottom - detection.top) * inner,
                    border=ft.Border.all(_BOX_WIDTH, color),
                    alignment=ft.Alignment.TOP_LEFT,
                    animate=_BOX_ANIMATION,
                    animate_position=_BOX_ANIMATION,
                    content=ft.Container(
                        bgcolor=color,
                        padding=ft.Padding.symmetric(horizontal=6.0, vertical=2.0),
                        animate=_BOX_ANIMATION,
                        content=ft.Text(
                            f"{query.strip()}: {score:.2f}",
                            color=ft.Colors.WHITE,
                            size=16.0,
                            weight=ft.FontWeight.BOLD,
                            max_lines=1,
                            overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                    ),
                )
            )
        if show_inference_times:
            sam_time, clip_time = inference_times
            overlay.append(
                ft.Container(
                    key="inference-times",
                    left=8.0,
                    bottom=8.0,
                    bgcolor=ft.Colors.BLACK_54,
                    padding=ft.Padding.symmetric(horizontal=6.0, vertical=2.0),
                    content=ft.Column(
                        tight=True,
                        spacing=0.0,
                        controls=[
                            ft.Text(f"FastSAM: {_format_inference_time(sam_time)}", color=ft.Colors.WHITE),
                            ft.Text(f"MobileCLIP: {_format_inference_time(clip_time)}", color=ft.Colors.WHITE),
                        ],
                    ),
                )
            )
        preview = ft.Stack(fit=ft.StackFit.EXPAND, controls=overlay)
    feed = ft.Container(
        width=side,
        height=side,
        border=ft.Border.all(_BORDER_WIDTH, _BORDER_COLOR),
        content=preview,
    )

    # A baseline has no effect without a main query, so editing one alone leaves nothing to submit.
    baseline_pending = bool(draft.strip()) and baseline_draft.strip() != baseline.strip()
    pending = draft.strip() != query.strip() or baseline_pending

    status: str | None
    if save_error is not None:
        status = save_error
    elif model_error is not None:
        status = model_error
    elif not model_ready:
        status = "Loading model…"
    elif pending:
        status = "Press Enter or tap the arrow to search."
    elif score is None or detection is None:
        status = None
    else:
        status = f"{'Detected' if detection.is_match(threshold) else 'Similarity'}: {score:.2f}"
        if detection.baseline_similarity is not None:
            status += f" (baseline: {detection.baseline_similarity:.2f})"

    def submit():
        set_query(draft)
        set_baseline(baseline_draft)

    def search_pressed():
        if pending:
            submit()
            set_search_generation(lambda n: n + 1)
        else:
            set_draft("")
            set_query("")
            set_baseline_draft("")
            set_baseline("")

    searching = bool(query.strip()) and not pending

    search = ft.Container(
        padding=ft.Padding.only(left=16.0, right=16.0, top=12.0),
        content=ft.Column(
            spacing=8.0,
            controls=[
                ft.TextField(
                    key=f"search-{search_generation}",
                    width=float("inf"),
                    value=draft,
                    on_change=lambda e: set_draft(e.control.value),
                    on_submit=lambda e: submit(),
                    suffix_icon=ft.IconButton(
                        icon=ft.Icons.CLOSE if searching else ft.Icons.ARROW_FORWARD,
                        icon_color=ft.Colors.WHITE,
                        tooltip="Clear search" if searching else "Search",
                        disabled=model_error is not None or not (pending or searching),
                        on_click=search_pressed,
                    ),
                    hint_text="What are you looking for?",
                    prefix_icon=ft.Icons.SEARCH,
                    color=ft.Colors.WHITE,
                    text_size=18.0,
                    hint_style=ft.TextStyle(size=18.0, color=ft.Colors.WHITE_70),
                    border_color=ft.Colors.WHITE_70,
                    focused_border_color=ft.Colors.WHITE,
                    cursor_color=ft.Colors.WHITE,
                    capitalization=ft.TextCapitalization.NONE,
                    autocorrect=False,
                    disabled=model_error is not None,
                ),
                ft.TextField(
                    key=f"baseline-{search_generation}",
                    width=float("inf"),
                    value=baseline_draft,
                    on_change=lambda e: set_baseline_draft(e.control.value),
                    on_submit=lambda e: submit(),
                    hint_text="Must score higher than… (optional)",
                    prefix_icon=ft.Icons.COMPARE_ARROWS,
                    dense=True,
                    color=ft.Colors.WHITE,
                    text_size=16.0,
                    hint_style=ft.TextStyle(size=16.0, color=ft.Colors.WHITE_70),
                    border_color=ft.Colors.WHITE_38,
                    focused_border_color=ft.Colors.WHITE,
                    cursor_color=ft.Colors.WHITE,
                    capitalization=ft.TextCapitalization.NONE,
                    autocorrect=False,
                    disabled=model_error is not None,
                ),
            ],
        ),
    )

    def slider_threshold(e: ft.Event[ft.Slider]) -> float:
        return round(e.control.value or _DETECTION_THRESHOLD, 2)

    async def save_threshold(e: ft.Event[ft.Slider]):
        try:
            await asyncio.to_thread(recordings_db.set_setting, _THRESHOLD_SETTING, f"{slider_threshold(e):.2f}")
        except Exception:
            pass

    threshold_slider = ft.Container(
        padding=ft.Padding.symmetric(horizontal=16.0),
        content=ft.Row(
            spacing=8.0,
            controls=[
                ft.Text("Threshold", color=ft.Colors.WHITE, size=18.0),
                ft.Slider(
                    key="threshold",
                    expand=True,
                    min=_MIN_THRESHOLD,
                    max=_MAX_THRESHOLD,
                    divisions=round((_MAX_THRESHOLD - _MIN_THRESHOLD) * 100),
                    value=threshold,
                    active_color=ft.Colors.WHITE,
                    inactive_color=ft.Colors.WHITE_38,
                    thumb_color=ft.Colors.WHITE,
                    on_change=lambda e: set_threshold(slider_threshold(e)),
                    on_change_end=save_threshold,
                ),
                ft.Text(f"{threshold:.2f}", color=ft.Colors.WHITE, size=18.0),
            ],
        ),
    )

    feed_area = ft.Container(
        expand=True,
        on_size_change=lambda e: set_area_size((e.width, e.height)),
        content=ft.Column(
            spacing=0.0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Container(
                    expand=True,
                    alignment=ft.Alignment.CENTER,
                    padding=ft.Padding.symmetric(horizontal=16.0),
                    content=ft.Text(
                        status or "",
                        color=ft.Colors.WHITE,
                        size=18.0,
                        text_align=ft.TextAlign.CENTER,
                        visible=status is not None,
                    ),
                ),
                feed,
                ft.Container(expand=True, alignment=ft.Alignment.CENTER, content=threshold_slider),
            ],
        ),
    )

    stop = ft.BottomAppBar(
        height=140.0,
        bgcolor=ft.Colors.GREY_900,
        elevation=0.0,
        content=record_button(
            label="Stop Recording",
            recording=True,
            on_click=stop_recording,
            key="stop",
            color=ft.Colors.WHITE,
            disabled=saving,
        ),
    )

    controls: list[ft.Control] = [
        ft.SafeArea(
            expand=True,
            content=ft.Column(
                expand=True,
                spacing=0.0,
                controls=[search, feed_area],
            ),
        )
    ]
    if supported:
        controls.append(
            fc.Camera(
                ref=camera,
                preview_enabled=False,
                width=0.0,
                height=0.0,
                on_stream_image=on_stream_image,
            )
        )

    return ft.View(
        route="/recording",
        padding=0.0,
        bgcolor=ft.Colors.BLACK,
        can_pop=False,
        appbar=ft.AppBar(
            title=ft.Text("Recording"),
            automatically_imply_leading=False,
            bgcolor=ft.Colors.GREY_900,
            color=ft.Colors.WHITE,
        ),
        bottom_appbar=stop,
        controls=[ft.Stack(expand=True, controls=controls)],
    )


def _load_threshold() -> float:
    try:
        threshold = float(recordings_db.get_setting(_THRESHOLD_SETTING, str(_DETECTION_THRESHOLD)))
    except ValueError:
        return _DETECTION_THRESHOLD
    return min(max(threshold, _MIN_THRESHOLD), _MAX_THRESHOLD)


def _format_inference_time(seconds: float | None) -> str:
    return "—" if seconds is None else f"{seconds * 1000.0:.0f} ms"


def _prepare_frame(jpeg: bytes, sensor_orientation: int):
    square = crop_square(jpeg, sensor_orientation)
    if square is None:
        return None, None
    return square, encode_jpeg(square)
