from tkinter import *
import tkinter
import logging
import somatictrainer.app

logging.basicConfig(level=logging.DEBUG)


def log_error(*args):
    logging.exception('whoops')


tkinter.Tk.report_callback_exception = log_error


def _main():
    root = Tk()
    # root.attributes('-topmost', 1)

    window = somatictrainer.app.SomaticTrainerHomeWindow(root)
    window.start()

    root.protocol('WM_DELETE_WINDOW', window.stop)
    while True:
        try:
            root.mainloop()
            break
        except UnicodeDecodeError:
            print("Caught inertial scrolling bug..?")


if __name__ == "__main__":
    _main()
