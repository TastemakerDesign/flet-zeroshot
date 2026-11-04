import flet as ft


def record_button(
    label: str,
    recording: bool,
    on_click,
    key: str,
    color: ft.ColorValue,
    disabled: bool,
) -> ft.Control:
    size = 30.0 if recording else 54.0
    return ft.Column(
        tight=True,
        spacing=8.0,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        opacity=0.4 if disabled else 1.0,
        controls=[
            ft.Container(
                key=key,
                width=72.0,
                height=72.0,
                shape=ft.BoxShape.CIRCLE,
                border=ft.Border.all(4.0, color),
                alignment=ft.Alignment.CENTER,
                tooltip=label,
                on_click=None if disabled else on_click,
                content=ft.Container(
                    width=size,
                    height=size,
                    bgcolor=ft.Colors.RED,
                    shape=ft.BoxShape.RECTANGLE if recording else ft.BoxShape.CIRCLE,
                ),
            ),
            ft.Text(label, size=18.0, weight=ft.FontWeight.BOLD, color=color),
        ],
    )
