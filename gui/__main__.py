"""Entry point: python -m gui"""
import sys

from gui.qt import QtWidgets
from gui.main_window import MainWindow, APP_NAME


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)

    # Runs the checks that only fail once packaged (see gui/self_test.py) and
    # exits. CI runs this against the built exe.
    if "--self-test" in argv:
        from gui.self_test import run

        return run()

    app = QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
