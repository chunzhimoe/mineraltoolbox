import multiprocessing
import sys
import tempfile
import traceback
from pathlib import Path


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
        report = Path(sys.argv[2])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text('{"status": "started", "stage": "importing desktop_check"}', encoding='utf-8')
        from geo_toolbox.desktop_check import run
        run(report)
        return
    from geo_toolbox.gui import main as gui_main
    gui_main()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        details = traceback.format_exc()
        path = Path(tempfile.gettempdir()) / 'mineraltoolbox-startup-error.txt'
        path.write_text(details, encoding='utf-8')
        if '--self-test' in sys.argv:
            if sys.stderr is not None:
                sys.stderr.write(details)
                sys.stderr.flush()
            raise SystemExit(1)
        import tkinter.messagebox
        tkinter.messagebox.showerror('MineralToolbox', f'启动失败，详细日志：{path}\n{details}')
        raise SystemExit(1)
