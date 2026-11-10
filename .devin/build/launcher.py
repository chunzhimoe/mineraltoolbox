import multiprocessing
import sys
import tempfile
import traceback
from pathlib import Path


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
        from geo_toolbox.desktop_check import run
        run(Path(sys.argv[2]))
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
        if '--self-test' not in sys.argv:
            import tkinter.messagebox
            tkinter.messagebox.showerror('MineralToolbox', f'启动失败，详细日志：{path}\n{details}')
        raise
