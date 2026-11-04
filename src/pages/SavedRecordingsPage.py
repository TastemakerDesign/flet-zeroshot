import asyncio
from typing import cast

import flet as ft

from components.confirm_delete import confirm_delete
from components.record_button import record_button
from utils.format_duration import format_duration
from utils.format_timestamp import format_timestamp
from utils.RecordingsDB import Recording, recordings_db


@ft.component
def SavedRecordingsPage():
    page = ft.context.page
    recordings, set_recordings = ft.use_state(cast(list[Recording] | None, None))
    error, set_error = ft.use_state(cast(str | None, None))

    async def load_recordings():
        try:
            set_recordings(await asyncio.to_thread(recordings_db.list_recordings))
        except Exception as e:
            set_error(f"Could not load recordings: {e}")

    location = ft.use_route_location()
    ft.use_effect(load_recordings, dependencies=[location])

    def on_deleted(recording_id: int):
        set_recordings(lambda rs: [r for r in rs or [] if r.id != recording_id])

    def confirm_swipe(e: ft.DismissibleDismissEvent, recording_id: int):
        dismissible = e.control

        async def deleted():
            await dismissible.confirm_dismiss(True)

        async def cancelled():
            await dismissible.confirm_dismiss(False)

        confirm_delete(recording_id, deleted, set_error, cancelled)

    def tile(r: Recording) -> ft.Control:
        return ft.Dismissible(
            key=str(r.id),
            dismiss_direction=ft.DismissDirection.END_TO_START,
            background=ft.Container(
                bgcolor=ft.Colors.ERROR,
                alignment=ft.Alignment.CENTER_RIGHT,
                padding=ft.Padding.only(right=24.0),
                content=ft.Row(
                    tight=True,
                    controls=[
                        ft.Icon(ft.Icons.DELETE_OUTLINE, color=ft.Colors.ON_ERROR),
                        ft.Text("Delete", color=ft.Colors.ON_ERROR, weight=ft.FontWeight.BOLD),
                    ],
                ),
            ),
            on_confirm_dismiss=lambda e, rid=r.id: confirm_swipe(e, rid),
            on_dismiss=lambda e, rid=r.id: on_deleted(rid),
            content=ft.ListTile(
                leading=ft.Icon(ft.Icons.VIDEOCAM),
                title=ft.Text(
                    format_timestamp(r.started_at),
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                subtitle=ft.Column(
                    tight=True,
                    spacing=4.0,
                    controls=[
                        ft.Text(f"Duration {format_duration(r.ended_at - r.started_at)}"),
                        _keyword_chips(r.prompts),
                    ],
                ),
                is_three_line=True,
                title_alignment=ft.ListTileTitleAlignment.TOP,
                trailing=ft.PopupMenuButton(
                    icon=ft.Icons.MORE_VERT,
                    tooltip="More options",
                    items=[
                        ft.PopupMenuItem(
                            content="Delete",
                            icon=ft.Icons.DELETE_OUTLINE,
                            on_click=lambda e, rid=r.id: confirm_delete(rid, lambda: on_deleted(rid), set_error, None),
                        )
                    ],
                ),
                on_click=lambda e, rid=r.id: page.navigate(f"/recordings/{rid}"),
            ),
        )

    if error is not None:
        body = ft.Container(
            alignment=ft.Alignment.CENTER,
            padding=24.0,
            content=ft.Text(error, color=ft.Colors.ERROR, text_align=ft.TextAlign.CENTER),
        )
    elif recordings is None:
        body = ft.Container(alignment=ft.Alignment.CENTER, content=ft.ProgressRing())
    elif not recordings:
        body = ft.Container(
            alignment=ft.Alignment.CENTER,
            padding=24.0,
            content=ft.Text(
                "No recordings yet. Start one to see it here.",
                text_align=ft.TextAlign.CENTER,
                size=20.0,
            ),
        )
    else:
        body = ft.ListView(
            padding=ft.Padding.symmetric(vertical=12.0),
            controls=[tile(r) for r in recordings],
        )

    return ft.View(
        route="/recordings",
        padding=0.0,
        appbar=ft.AppBar(
            title=ft.Text("Saved Recordings"),
            automatically_imply_leading=False,
            actions=[
                ft.IconButton(
                    icon=ft.Icons.SETTINGS_OUTLINED,
                    tooltip="Settings",
                    on_click=lambda: page.navigate("/recordings/settings"),
                )
            ],
        ),
        bottom_appbar=ft.BottomAppBar(
            height=140.0,
            bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            elevation=0.0,
            content=ft.Container(
                alignment=ft.Alignment.CENTER,
                content=record_button(
                    label="Start Recording",
                    recording=False,
                    on_click=lambda: page.navigate("/recording"),
                    key="new",
                    color=ft.Colors.ON_SURFACE,
                    disabled=False,
                ),
            ),
        ),
        controls=[
            ft.SafeArea(
                expand=True,
                content=ft.Column(
                    expand=True,
                    spacing=0.0,
                    controls=[
                        ft.Container(
                            expand=True,
                            content=body,
                        ),
                    ],
                ),
            )
        ],
    )


_KEYWORD_COLOR = ft.Colors.with_opacity(0.8, ft.Colors.ON_SURFACE)


def _keyword_chips(prompts: list[str]) -> ft.Control:
    if not prompts:
        return ft.Text("No keywords searched", italic=True, color=_KEYWORD_COLOR)
    return ft.Row(
        spacing=6.0,
        scroll=ft.ScrollMode.HIDDEN,
        controls=[
            ft.Container(
                key=prompt,
                padding=ft.Padding.symmetric(horizontal=12.0, vertical=4.0),
                border=ft.Border.all(1.0, _KEYWORD_COLOR),
                border_radius=8.0,
                content=ft.Text(prompt, size=15.0, color=_KEYWORD_COLOR),
            )
            for prompt in prompts
        ],
    )
