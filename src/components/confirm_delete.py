import asyncio
import inspect

import flet as ft

from utils.RecordingsDB import recordings_db


def confirm_delete(recording_id: int, on_deleted, on_error, on_cancel):
    page = ft.context.page
    decided = False

    async def call(callback, *args):
        if callback is not None and inspect.isawaitable(result := callback(*args)):
            await result

    async def delete():
        nonlocal decided
        decided = True
        page.pop_dialog()
        try:
            await asyncio.to_thread(recordings_db.delete_recording, recording_id)
        except Exception as e:
            await call(on_cancel)
            await call(on_error, f"Could not delete the recording: {e}")
            return
        await call(on_deleted)

    async def cancel():
        nonlocal decided
        if decided:
            return
        decided = True
        page.pop_dialog()
        await call(on_cancel)

    page.show_dialog(
        ft.AlertDialog(
            title=ft.Text("Delete recording?"),
            content=ft.Text("This permanently deletes the recording and all of its similarity measurements."),
            actions=[
                ft.TextButton(content="Cancel", on_click=cancel),
                ft.TextButton(
                    content="Delete",
                    style=ft.ButtonStyle(color=ft.Colors.ERROR),
                    on_click=delete,
                ),
            ],
            on_dismiss=cancel,
        )
    )
