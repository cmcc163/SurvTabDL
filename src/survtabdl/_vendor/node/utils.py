"""Minimal tensor conversion needed by NODE initialization."""
def check_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else value
