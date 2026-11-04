from typing import cast

import flet as ft
import flet_permission_handler as fph


@ft.component
def CameraPermissionPage():
    error, set_error = ft.use_state(cast(str | None, None))
    handler = ft.use_ref(None)

    def get_handler() -> fph.PermissionHandler:
        if handler.current is None:
            handler.current = fph.PermissionHandler()
        return handler.current

    async def request_permission():
        try:
            status = await get_handler().request(fph.Permission.CAMERA)
        except ft.FletUnsupportedPlatformException:
            set_error("Camera access isn't supported on this platform. Run the app on Android or iOS.")
            return
        if status == fph.PermissionStatus.GRANTED:
            set_error(None)
            ft.context.page.navigate("/recordings")
        else:
            set_error(_DENIED_INSTRUCTIONS)

    async def open_settings():
        await get_handler().open_app_settings()

    controls = [
        ft.Icon(ft.Icons.PHOTO_CAMERA, size=96.0),
        ft.Text(
            "We need access to your camera to show the live feed.",
            text_align=ft.TextAlign.CENTER,
            size=20.0,
        ),
        ft.FilledButton(
            content="Allow camera access",
            icon=ft.Icons.CAMERA_ALT,
            key="request",
            on_click=request_permission,
        ),
    ]
    if error is not None:
        controls.append(
            ft.Container(
                key="error",
                padding=16.0,
                bgcolor=ft.Colors.ERROR_CONTAINER,
                content=ft.Column(
                    spacing=12.0,
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Icon(
                                    ft.Icons.ERROR_OUTLINE,
                                    color=ft.Colors.ON_ERROR_CONTAINER,
                                ),
                                ft.Text(
                                    "Permission not granted",
                                    weight=ft.FontWeight.BOLD,
                                    color=ft.Colors.ON_ERROR_CONTAINER,
                                ),
                            ],
                        ),
                        ft.Text(error, color=ft.Colors.ON_ERROR_CONTAINER),
                        ft.TextButton(
                            content="Open settings",
                            key="settings",
                            on_click=open_settings,
                            visible=error == _DENIED_INSTRUCTIONS,
                        ),
                    ],
                ),
            )
        )

    return ft.View(
        route="/permission",
        appbar=ft.AppBar(title=ft.Text("Camera access")),
        controls=[
            ft.SafeArea(
                expand=True,
                content=ft.Container(
                    padding=24.0,
                    content=ft.Column(
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=24.0,
                        controls=controls,
                    ),
                ),
            )
        ],
    )


_DENIED_INSTRUCTIONS = (
    "Camera access is required to continue. To grant it:\n\n"
    "- iOS: open Settings, find this app, and turn on Camera.\n"
    "- Android: open Settings > Apps, choose this app, tap Permissions, and allow Camera.\n"
)
