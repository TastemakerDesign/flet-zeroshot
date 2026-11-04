import flet_permission_handler as fph


async def has_camera_permission() -> bool:
    try:
        status = await fph.PermissionHandler().get_status(fph.Permission.CAMERA)
    except Exception:
        return False
    return status == fph.PermissionStatus.GRANTED
