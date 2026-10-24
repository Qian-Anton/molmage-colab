"""Small output helpers shared by the notebook-facing functions."""


def _ipython():
    try:
        from IPython import get_ipython

        return get_ipython()
    except ImportError:
        return None


class Status:
    """One line of text that is rewritten in place (plain print outside IPython)."""

    def __init__(self, text=""):
        self._handle = None
        if _ipython() is not None:
            from IPython.display import Pretty, display

            self._handle = display(Pretty(text), display_id=True)
        else:
            print(text, flush=True)

    def update(self, text):
        if self._handle is not None:
            from IPython.display import Pretty

            self._handle.update(Pretty(text))
        else:
            print(text, flush=True)


def note(text):
    """A single line of output (same look as Status)."""
    if _ipython() is not None:
        from IPython.display import Pretty, display

        display(Pretty(text))
    else:
        print(text, flush=True)
