import pytest

tk = pytest.importorskip("tkinter")

from geo_toolbox import gui, geochem_gui
from tk_test_support import create_tk_root


@pytest.mark.parametrize("workbench", [gui.MagneticWorkbench, geochem_gui.GeochemWorkbench])
def test_constructor_has_no_callback_errors_and_mapping_still_invalidates(monkeypatch, workbench):
    errors = []
    monkeypatch.setattr(gui.MagneticWorkbench, "_on_callback_error",
                        lambda self, kind, value, tb: errors.append(value))
    root = create_tk_root()
    root.withdraw()
    app = None
    try:
        app = workbench(root)
        root.update()
        assert errors == []
        assert app.grid_label.winfo_exists()
        assert app.log_text.winfo_exists()
        assert not app._populating
        assert app.skip_ids_var.get()
        if isinstance(app, geochem_gui.GeochemWorkbench):
            assert app.zero_var.get() == "待复核（不能直接填补）"
        app.dataset = object()
        app.last_report = {"previous": True}
        app.map_vars["x"].set("changed")
        assert app.dataset is None
        assert app.last_report is None
        assert errors == []
    finally:
        if app is not None:
            app._on_close()
        else:
            root.destroy()


@pytest.mark.parametrize("kind, label", [("magnetic", "磁测工作台"),
                                         ("geochem", "地球化学工作台")])
def test_main_chooser_opens_workbench_without_callback_errors(monkeypatch, kind, label):
    errors, opened = [], []
    magnetic = gui.MagneticWorkbench
    geochem = geochem_gui.GeochemWorkbench
    monkeypatch.setattr(magnetic, "_on_callback_error",
                        lambda self, error_type, value, tb: errors.append(value))
    root = create_tk_root()
    root.withdraw()
    root.report_callback_exception = lambda error_type, value, tb: errors.append(value)

    def build(workbench):
        app = workbench(root)
        opened.append(app)
        return app

    def interact():
        chooser = next(widget for widget in root.winfo_children()
                       if widget.winfo_class() == "TFrame")
        button = next(widget for widget in chooser.winfo_children()
                      if widget.winfo_class() == "TButton" and widget.cget("text") == label)
        button.invoke()
        root.update()
        assert errors == []
        assert len(opened) == 1
        app = opened[0]
        assert isinstance(app, geochem if kind == "geochem" else magnetic)
        assert app.grid_label.winfo_exists()
        assert app.log_text.winfo_exists()
        assert app.notebook.select() == str(app.tab_import)

    monkeypatch.setattr(tk, "Tk", lambda: root)
    monkeypatch.setattr(root, "mainloop", interact)
    monkeypatch.setattr(gui, "MagneticWorkbench", lambda parent: build(magnetic))
    monkeypatch.setattr(geochem_gui, "GeochemWorkbench", lambda parent: build(geochem))
    try:
        gui.main([])
    finally:
        if opened:
            opened[0]._on_close()
        else:
            root.destroy()


@pytest.mark.parametrize("workbench", [gui.MagneticWorkbench, geochem_gui.GeochemWorkbench])
def test_frozen_workbench_check_rejects_callback_errors(monkeypatch, workbench):
    from geo_toolbox.desktop_check import _workbench_checks

    build_import = workbench._build_import

    def broken_import(self):
        build_import(self)
        var = tk.StringVar(master=self.root)

        def fail(*args):
            raise ValueError("injected startup callback failure")

        var.trace_add("write", fail)
        var.set("trigger")

    monkeypatch.setattr(workbench, "_build_import", broken_import)
    with pytest.raises(RuntimeError, match="injected startup callback failure"):
        _workbench_checks()
