from tkinter import *
import logging
import somatictrainer.app

logging.basicConfig(level=logging.INFO)

def _main():
    root = Tk()
    # root.withdraw()
    root.attributes('-topmost', 1)

    window = somatictrainer.app.SomaticTrainerHomeWindow(root)
    window.start()

    root.protocol('WM_DELETE_WINDOW', window.stop)
    root.mainloop()


if __name__ == "__main__":
    _main()
