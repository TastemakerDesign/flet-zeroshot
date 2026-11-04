import asyncio

import flet as ft

from pages.CameraPermissionPage import CameraPermissionPage
from pages.OnboardingPage import OnboardingPage
from pages.RecordingDetailsPage import RecordingDetailsPage
from pages.RecordingPage import RecordingPage
from pages.SavedRecordingsPage import SavedRecordingsPage
from pages.SettingsPage import SettingsPage
from utils.has_camera_permission import has_camera_permission
from utils.SettingsManager import settings_manager


def _theme(on_surface: str) -> ft.Theme:
    return ft.Theme(
        text_theme=ft.TextTheme(
            body_large=ft.TextStyle(size=19.0),
            body_medium=ft.TextStyle(size=17.0),
            body_small=ft.TextStyle(size=15.0),
            label_large=ft.TextStyle(size=17.0),
            label_medium=ft.TextStyle(size=15.0),
            label_small=ft.TextStyle(size=13.0),
            title_large=ft.TextStyle(size=26.0),
            title_medium=ft.TextStyle(size=19.0),
            title_small=ft.TextStyle(size=17.0),
        ),
        color_scheme=ft.ColorScheme(on_surface=on_surface, on_surface_variant=on_surface),
        appbar_theme=ft.AppBarTheme(
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST, elevation=0.0, elevation_on_scroll=0.0
        ),
    )


async def _get_launch_route(route: str) -> str:
    if route == "/" and await asyncio.to_thread(settings_manager.get_onboarded):
        route = "/permission"
    if route == "/permission" and await has_camera_permission():
        route = "/recordings"
    return route


@ft.component
def _App():
    page = ft.context.page
    launched, set_launched = ft.use_state(False)

    async def launch():
        route = await _get_launch_route(page.route or "/")
        page.route = route
        await page.push_route(route)
        set_launched(True)

    ft.on_mounted(launch)

    if not launched:
        return [ft.View(route=page.route or "/", controls=[])]

    return ft.Router(
        [
            ft.Route(index=True, component=OnboardingPage),
            ft.Route(path="permission", component=CameraPermissionPage),
            ft.Route(
                path="recordings",
                component=SavedRecordingsPage,
                # `settings` comes first so it isn't matched as a recording id.
                children=[
                    ft.Route(path="settings", component=SettingsPage),
                    ft.Route(path=":id", component=RecordingDetailsPage),
                ],
            ),
            ft.Route(path="recording", component=RecordingPage),
        ],
        manage_views=True,
    )


def main(page: ft.Page):
    page.theme = _theme(ft.Colors.BLACK)
    page.dark_theme = _theme(ft.Colors.WHITE)
    return page.render_views(_App)


if __name__ == "__main__":
    ft.run(main)
