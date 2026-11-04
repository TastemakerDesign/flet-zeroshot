import flet as ft


def is_camera_supported(page: ft.Page) -> bool:
    return page.platform in (ft.PagePlatform.ANDROID, ft.PagePlatform.IOS)
