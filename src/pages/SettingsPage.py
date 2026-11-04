import asyncio
from typing import cast

import flet as ft
import flet_camera as fc

from utils.create_session import available_providers
from utils.ModelPrecision import ModelPrecision
from utils.SettingsManager import MODEL_PRECISIONS, RESOLUTIONS, settings_manager


@ft.component
def SettingsPage():
    page = ft.context.page
    resolution, set_resolution = ft.use_state(cast(fc.ResolutionPreset | None, None))
    vibration, set_vibration = ft.use_state(True)
    precision, set_precision = ft.use_state(cast(ModelPrecision | None, None))
    inference_overlay, set_inference_overlay = ft.use_state(False)
    error, set_error = ft.use_state(cast(str | None, None))

    async def load():
        try:
            set_vibration(await asyncio.to_thread(settings_manager.get_vibration))
            set_inference_overlay(await asyncio.to_thread(settings_manager.get_inference_overlay))
            set_precision(await asyncio.to_thread(settings_manager.get_model_precision))
            set_resolution(await asyncio.to_thread(settings_manager.get_resolution))
        except Exception as e:
            set_error(f"Could not load the settings: {e}")

    ft.use_effect(load, dependencies=[])

    async def change_resolution(e: ft.Event[ft.RadioGroup]):
        selected = fc.ResolutionPreset(e.control.value)
        set_resolution(selected)
        try:
            await asyncio.to_thread(settings_manager.set_resolution, selected)
        except Exception as ex:
            set_error(f"Could not save the setting: {ex}")

    async def change_vibration(e: ft.Event[ft.RadioGroup]):
        enabled = e.control.value == "on"
        set_vibration(enabled)
        try:
            await asyncio.to_thread(settings_manager.set_vibration, enabled)
        except Exception as ex:
            set_error(f"Could not save the setting: {ex}")

    async def change_precision(e: ft.Event[ft.RadioGroup]):
        selected = ModelPrecision(e.control.value)
        set_precision(selected)
        try:
            await asyncio.to_thread(settings_manager.set_model_precision, selected)
        except Exception as ex:
            set_error(f"Could not save the setting: {ex}")

    async def change_inference_overlay(e: ft.Event[ft.Switch]):
        enabled = bool(e.control.value)
        set_inference_overlay(enabled)
        try:
            await asyncio.to_thread(settings_manager.set_inference_overlay, enabled)
        except Exception as ex:
            set_error(f"Could not save the setting: {ex}")

    body: list[ft.Control]
    if error is not None:
        body = [ft.Text(error, color=ft.Colors.ERROR)]
    elif resolution is None or precision is None:
        body = [ft.Container(alignment=ft.Alignment.CENTER, content=ft.ProgressRing())]
    else:
        body = [
            ft.Text("Camera Resolution", size=20.0, weight=ft.FontWeight.BOLD),
            ft.RadioGroup(
                value=resolution.value,
                on_change=change_resolution,
                content=ft.Column(
                    spacing=4.0,
                    controls=[
                        ft.Radio(key=preset.value, value=preset.value, label=name)
                        for preset, name in RESOLUTIONS.items()
                    ],
                ),
            ),
            ft.Container(
                padding=ft.Padding.only(top=16.0),
                content=ft.Text("Vibration", size=20.0, weight=ft.FontWeight.BOLD),
            ),
            ft.RadioGroup(
                key="vibration",
                value="on" if vibration else "off",
                on_change=change_vibration,
                content=ft.Column(
                    spacing=4.0,
                    controls=[
                        ft.Radio(key="on", value="on", label="On"),
                        ft.Radio(key="off", value="off", label="Off"),
                    ],
                ),
            ),
            ft.Container(
                padding=ft.Padding.only(top=16.0),
                content=ft.Text("Model Precision", size=20.0, weight=ft.FontWeight.BOLD),
            ),
            ft.RadioGroup(
                key="model-precision",
                value=precision.value,
                on_change=change_precision,
                content=ft.Column(
                    spacing=4.0,
                    controls=[
                        ft.Radio(key=option.value, value=option.value, label=name)
                        for option, name in MODEL_PRECISIONS.items()
                    ],
                ),
            ),
            ft.Container(
                padding=ft.Padding.only(top=16.0),
                content=ft.Text("Debugging", size=20.0, weight=ft.FontWeight.BOLD),
            ),
            ft.Switch(
                key="debug-inference-overlay",
                # Switch has no label spacing option, so the label is padded instead.
                label=ft.Container(padding=ft.Padding.only(left=12.0), content=ft.Text("Show Inference Times")),
                value=inference_overlay,
                on_change=change_inference_overlay,
            ),
            ft.OutlinedButton(
                content="Open Onboarding Page",
                key="debug-onboarding",
                on_click=lambda: page.navigate("/"),
            ),
            ft.OutlinedButton(
                content="Open Camera Permission Page",
                key="debug-permission",
                on_click=lambda: page.navigate("/permission"),
            ),
            ft.Container(
                padding=ft.Padding.only(top=8.0),
                content=ft.Text("ONNX Runtime Providers", weight=ft.FontWeight.BOLD),
            ),
            *[ft.Text(provider, selectable=True) for provider in available_providers()],
        ]

    return ft.View(
        route="/recordings/settings",
        appbar=ft.AppBar(title=ft.Text("Settings")),
        controls=[
            ft.SafeArea(
                expand=True,
                content=ft.ListView(
                    expand=True,
                    padding=ft.Padding.symmetric(horizontal=16.0, vertical=12.0),
                    spacing=8.0,
                    controls=body,
                ),
            )
        ],
    )
