import uuid


def get_unique_id() -> str:
    return str(uuid.uuid4())[:8]
