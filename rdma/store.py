# design/whole-function.md#observations
import contextlib

_state = {}


def load():
    return _state


@contextlib.contextmanager
def changing():
    yield _state


def records(section):
    """A section's records."""
    return load().get(section, {})
