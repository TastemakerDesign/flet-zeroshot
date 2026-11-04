import asyncio

import flet as ft

from utils.SettingsManager import settings_manager


@ft.component
def OnboardingPage():
    page = ft.context.page

    async def next_page():
        try:
            await asyncio.to_thread(settings_manager.set_onboarded)
        except Exception:
            pass
        page.navigate("/permission")

    return ft.View(
        route=ft.use_route_location() or "/",
        appbar=ft.AppBar(title=ft.Text("Welcome")),
        controls=[
            ft.SafeArea(
                expand=True,
                content=ft.Container(
                    padding=24.0,
                    content=ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=24.0,
                        controls=[
                            ft.Image(src="icon.png", width=160, height=160),
                            ft.Text(
                                "This app uses your camera to search for anything you describe. Type a keyword and point your camera at the world around you.",
                                text_align=ft.TextAlign.CENTER,
                                size=20.0,
                            ),
                            ft.FilledButton(
                                content="Next",
                                key="next",
                                on_click=next_page,
                            ),
                        ],
                    ),
                ),
            )
        ],
    )
