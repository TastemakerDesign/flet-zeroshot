from datetime import datetime


def format_timestamp(ts: float) -> str:
    local = datetime.fromtimestamp(ts)
    hour = local.hour % 12 or 12
    return f"{local:%b} {local.day}, {local:%Y} · {hour}:{local:%M:%S %p}"
