import tkinter as tk

import pytest

import tk_test_support


def _patch(monkeypatch, behavior, platform="win32"):
    calls = []
    sleeps = []
    monkeypatch.setattr(tk_test_support.tk, "Tk", _run(behavior, calls))
    monkeypatch.setattr(tk_test_support.time, "sleep", sleeps.append)
    monkeypatch.setattr(tk_test_support.sys, "platform", platform)
    return calls, sleeps


def _run(behavior, calls):
    def wrapped():
        calls.append(None)
        return behavior(len(calls))
    return wrapped


def test_success_returns_immediately(monkeypatch):
    sentinel = object()
    calls, sleeps = _patch(monkeypatch, lambda n: sentinel)
    assert tk_test_support.create_tk_root() is sentinel
    assert len(calls) == 1 and sleeps == []


def test_matching_error_retries_once(monkeypatch, capsys):
    sentinel = object()
    message = ("Can't find a usable init.tcl\n"
               "couldn't read file \"x/init.tcl\": No error")

    def behavior(n):
        if n == 1:
            raise tk.TclError(message)
        return sentinel

    calls, sleeps = _patch(monkeypatch, behavior)
    assert tk_test_support.create_tk_root() is sentinel
    assert len(calls) == 2 and sleeps == [0.3]
    stderr = capsys.readouterr().err
    assert "attempt 1" in stderr
    assert message in stderr


def test_persistent_error_raises_original(monkeypatch):
    message = ("Can't find a usable init.tcl\n"
               "couldn't read file \"x/init.tcl\": No error")
    raised = []

    def behavior(n):
        error = tk.TclError(message)
        raised.append(error)
        raise error

    calls, sleeps = _patch(monkeypatch, behavior)
    with pytest.raises(tk.TclError) as exc:
        tk_test_support.create_tk_root()
    assert exc.value is raised[-1]
    assert len(calls) == 5 and sleeps == [0.3] * 4


@pytest.mark.parametrize("message", [
    "Can't find a usable init.tcl\nno such file or directory",
    "couldn't connect to display",
    "arbitrary Tcl failure",
    "Can't find a usable init.tcl\ncouldn't read file \"x\": access denied",
])
def test_non_matching_tclerror_propagates(monkeypatch, message):
    error = tk.TclError(message)

    def behavior(n):
        raise error

    calls, sleeps = _patch(monkeypatch, behavior)
    with pytest.raises(tk.TclError) as exc:
        tk_test_support.create_tk_root()
    assert exc.value is error
    assert len(calls) == 1 and sleeps == []


def test_non_tclerror_propagates(monkeypatch):
    error = ValueError("boom")

    def behavior(n):
        raise error

    calls, sleeps = _patch(monkeypatch, behavior)
    with pytest.raises(ValueError) as exc:
        tk_test_support.create_tk_root()
    assert exc.value is error
    assert len(calls) == 1 and sleeps == []


def test_matching_error_non_windows_no_retry(monkeypatch):
    message = ("Can't find a usable init.tcl\n"
               "couldn't read file \"x/init.tcl\": No error")
    error = tk.TclError(message)

    def behavior(n):
        raise error

    calls, sleeps = _patch(monkeypatch, behavior, platform="linux")
    with pytest.raises(tk.TclError) as exc:
        tk_test_support.create_tk_root()
    assert exc.value is error
    assert len(calls) == 1 and sleeps == []
