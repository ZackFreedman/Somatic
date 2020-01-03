from typing import List

from tkinter import *
from tkinter import messagebox, filedialog, ttk
import serial
from serial import Serial, SerialException
from serial.tools.list_ports import comports
import logging
import numpy as np
from pyquaternion import Quaternion
from threading import Thread
from queue import Queue, Empty
import json
import os
from PIL import Image, ImageTk

logging.basicConfig(level=logging.DEBUG)

root = Tk()
root.attributes('-topmost', 1)

standard_gesture_time = 1000  # Milliseconds
sampling_rate = 10  # Milliseconds

hand_icon_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Hands')
# hand_icons = {i: PhotoImage(file=os.path.join(hand_icon_directory,'{:04b}.png'.format(i))) for i in range(16)}

unknown_hand_icon_bitmap = Image.open(os.path.join(hand_icon_directory, 'Unknown.png'))
unknown_hand_icon_bitmap.thumbnail((250, 250))
unknown_hand_icon = ImageTk.PhotoImage(image=unknown_hand_icon_bitmap)

hand_bitmaps = []
hand_icons = {}

for i in range(16):
    hand_bitmap = Image.open(os.path.join(hand_icon_directory,'{:04b}.png'.format(i)))
    hand_bitmap.thumbnail((250, 250))

    hand_bitmaps.append(hand_bitmap)
    hand_icons[i] = ImageTk.PhotoImage(image=hand_bitmap)

class Gesture:
    def __init__(self, glyph, raw_data, normalized_quats=None):
        """

        :param normalized_quats:
        :type normalized_quats: list of Quaternion or None
        :param raw_data:
        :type raw_data tuple of list of Quaternion, float or None
        :param glyph:
        :type glyph: str
        """
        if normalized_quats is not None and not len(normalized_quats) is 100:
            raise AttributeError('Normalized data invalid - got {} normalized_data instead of {}'
                                 .format(len(normalized_quats), standard_gesture_time / sampling_rate))

        if raw_data is None:
            raise AttributeError('Must provide one source of data')

        if normalized_quats is None:
            normalized_quats = Gesture.normalize_samples(raw_data[0], raw_data[1])

        self.raw_quats = raw_data[0]
        self.raw_timedeltas = raw_data[1]
        self.normalized_data = normalized_quats
        self.glyph = glyph

    def to_dict(self):
        datastore = {
            'g': self.glyph,
            'r': [(s.w, s.x, s.y, s.z, ts) for s, ts in zip(self.raw_quats, self.raw_timedeltas)],
            'n': [(s.w, s.x, s.y, s.z) for s in self.normalized_data]
        }
        return datastore

    @staticmethod
    def from_dict(datastore):
        try:
            glyph = datastore['g']

            raw_data = [(Quaternion(w=e[0], x=e[1], y=e[2], z=e[3]), e[4]) for e in datastore['r']]
            normalized_quats = [Quaternion(w=e[0], x=e[1], y=e[2], z=e[3]) for e in datastore['n']]
            assert len(normalized_quats) is round(standard_gesture_time / sampling_rate)

            return Gesture(glyph, raw_data, normalized_quats)

        except (AssertionError, AttributeError, KeyError):
            logging.exception('Error parsing dict {}...'.format(str(datastore)[:20]))

        return None

    @staticmethod
    def normalize_samples(samples: List[Quaternion], timedeltas):
        if not samples or not timedeltas:
            raise AttributeError('Samples and/or timedeltas not provided')

        if len(samples) != len(timedeltas):
            raise AttributeError('Samples and timedeltas must be the same length')

        if not len(samples) or not len(timedeltas):
            raise AttributeError('Samples and/or timedeltas list are empty')

        scaling_factor = standard_gesture_time / sum(timedeltas)
        # Standardize times to 1 second
        scaled_times = [delta * scaling_factor for delta in timedeltas]

        output = []

        # Interpolate to increase/reduce number of samples to required sampling rate
        for earliest_time in range(0, standard_gesture_time, sampling_rate):
            # For each sample required, find the latest sample before this time, and the earliest sample
            # after this time, and slerp them.

            early_sample = None
            early_time = None
            late_sample = None
            late_time = None

            latest_time = earliest_time + sampling_rate

            sample_time = 0
            for index, sample in enumerate(samples):
                if index:
                    sample_time += scaled_times[index]

                if early_sample is None and sample_time >= earliest_time:
                    # This sample is the latest sample that began earlier than the early time.
                    early_sample = samples[index]
                    early_time = sample_time - scaled_times[index]

                if late_sample is None and sample_time >= latest_time:
                    # This sample is the latest sample that began earlier than the late time.
                    late_sample = samples[index]
                    late_time = sample_time

                if early_sample and late_sample:
                    continue

            if not early_sample or not late_sample:
                raise AttributeError('Something went wrong - no samples work')

            amount = (earliest_time - early_time) / (late_time - early_time)  # Just the Arduino map function
            output.append(Quaternion.slerp(early_sample, late_sample, amount))

        return output


class GestureTrainingSet:
    big_ole_list_o_glyphs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!@#/ 1234567890'
    current_version = 1  # For deleting old saves

    def __init__(self):
        self.target_examples_per_glyph = 100

        self.examples = []

    @staticmethod
    def load(pathspec):
        with open(pathspec, 'r') as f:
            datastore = json.load(f)

            if 'version' not in datastore or datastore['version'] != GestureTrainingSet.current_version:
                logging.warning("Saved file is outdated, not loading")
                return

            output = GestureTrainingSet()

            for sample_record in datastore['examples']:
                output.add(Gesture.from_dict(sample_record))

            return output

    def save(self, pathspec):
        datastore = {'version': GestureTrainingSet.current_version, 'examples': [x.to_dict() for x in self.examples]}
        with open(pathspec, 'w') as f:
            json.dump(datastore, f)

    def add(self, example):
        self.examples.append(example)

    def count(self, glyph):
        return len([example for example in self.examples if example.glyph == glyph])

    def summarize(self):
        return {glyph: self.count(glyph) for glyph in self.big_ole_list_o_glyphs}


class SomaticTrainerHomeWindow(Frame):
    port: serial.Serial
    serial_sniffing_thread: Thread
    training_set: GestureTrainingSet

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        self.queue = Queue()
        self.port = None
        self.serial_sniffing_thread = None
        self.receiving = False

        self.open_file_pathspec = ''
        self.open_file_has_been_modified = False
        self.training_set = None

        self.master.title('Somatic Trainer')
        self.pack(fill=BOTH, expand=1)

        self.menu_bar = Menu(master)

        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label='New', command=self.new_file)
        file_menu.add_command(label='Open', command=self.open_file)
        file_menu.add_command(label='Save', command=self.save_file, state=DISABLED)
        file_menu.add_command(label='Save as...', command=self.save_as, state=DISABLED)
        self.menu_bar.add_cascade(label='File', menu=file_menu)

        self.port_menu = Menu(self.menu_bar, tearoff=0, postcommand=self.populate_serial_port_menu)
        self.menu_bar.add_cascade(label='Serial Port', menu=self.port_menu)

        master.config(menu=self.menu_bar)

        left_column = Frame(self, width=250, bg='red')
        right_column = Frame(self, bg='blue')

        Label(left_column, text='Hand').pack(fill=X)
        self.hand_display = Canvas(left_column, width=250, height=250)
        self.hand_display.create_image((0, 0), image=unknown_hand_icon, anchor=N+W)
        self.hand_display.pack(fill=X)

        Label(left_column, text='Path').pack(fill=X)
        self.path_display = Canvas(left_column, width=250, height=250)
        self.path_display.pack(fill=X)

        self.file_name_label = Label(right_column, text='No training file open', bg='white', relief=SUNKEN)
        self.file_name_label.pack(fill=X, anchor=N+E)

        self.glyph_picker = ttk.Treeview(right_column, column='count')
        self.glyph_picker.column("#0", width=100, stretch=False)
        self.glyph_picker.column('count', stretch=True)
        self.glyph_picker.heading('count', text='Count')
        self.glyph_picker.pack(fill=BOTH, expand=1, anchor=S+E)
        for glyph in GestureTrainingSet.big_ole_list_o_glyphs:
            self.glyph_picker.insert('', 'end', text="Glyph '{}'".format(glyph), value='0')

        self.status_line = Label(self, text='Not connected', bd=1, relief=SUNKEN, anchor=W, bg='light goldenrod')

        left_column.grid(row=0, column=0, sticky=N)
        right_column.grid(row=0, column=1, sticky=N+S+E+W)
        self.status_line.grid(row=1, column=0, columnspan=2, sticky=S+E+W)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def start(self):
        # self.reload_serial_port_picker()
        root.after(10, self.queue_handler)

    def stop(self):
        self.receiving = False
        self.queue.put({'type': 'quit'})

    def queue_handler(self):
        try:
            command = self.queue.get(block=False)
            if command['type'] is 'rx':
                self.update_status(command['fingers'],
                                   command['quat'][0], command['quat'][1], command['quat'][2], command['quat'][3],
                                   command['freq'])
                logging.info('Updating status')
            elif command['type'] is 'quit':
                return
        except Empty:
            pass

        root.after(10, self.queue_handler)

    def new_file(self):
        if self.open_file_has_been_modified:
            outcome = messagebox.askyesnocancel('Unsaved changes',
                                                'Training data has been changed.\nSave before creating new file?')
            if outcome:
                self.save_file()
            elif outcome is None:
                return

        self.training_set = GestureTrainingSet()
        self.open_file_pathspec = filedialog.asksaveasfilename(
            title='Create training file',
            filetypes=(('JSON dictionary', '*.json'), ('All files', '*')))
        self.open_file_has_been_modified = True

    def open_file(self):
        file_to_open = filedialog.askopenfilename(title='Select training data',
                                                  filetypes=(('JSON dictionary', '*.json'), ('All files', '*')))

        if file_to_open:
            try:
                training_set = GestureTrainingSet.load(file_to_open)
                self.training_set = training_set
                self.open_file_pathspec = file_to_open
                self.menu_bar.entryconfigure('Save', state=NORMAL)
                self.menu_bar.entryconfigure('Save as...', state=NORMAL)
                self.file_name_label.configure(text=self.open_file_pathspec)
            except (KeyError, AttributeError, json.decoder.JSONDecodeError) as e:
                logging.exception('Oopsie')
                messagebox.showerror("Can't load file",
                                     "This file can't be loaded.\nError: {}".format(repr(e)))

    def save_file(self):
        if not self.training_set:
            logging.warning('Tried to save empty training set?')
            return

        if not self.open_file_pathspec:
            file_to_create = filedialog.asksaveasfilename(title='Save training data',
                                                          filetypes=(('JSON dictionary', '*.json'), ('All files', '*')))
            if file_to_create:
                self.open_file_pathspec = file_to_create
            else:
                return

        self.training_set.save(self.open_file_pathspec)

        self.open_file_has_been_modified = False

    def save_as(self):
        if not self.training_set:
            logging.warning('Tried to save empty training set?')
            return

        file_to_create = filedialog.asksaveasfilename(title='Save training data',
                                                      filetypes=(('JSON dictionary', '*.json'), ('All files', '*')))
        if file_to_create:
            self.open_file_pathspec = file_to_create
        else:
            return

        self.training_set.save(self.open_file_pathspec)

        self.open_file_has_been_modified = False

    # def reload_serial_port_picker(self):
    #     self.serial_port_picker.delete(0, END)
    #     ports = comports()
    #     if ports:
    #         for path, _, __ in ports:
    #             self.serial_port_picker.insert(END, path)
    #             self.serial_port_picker.select_set(0)
    #             self.serial_connect_button.configure(state=NORMAL)
    #     else:
    #         self.serial_port_picker.insert('No serial ports!')

    def populate_serial_port_menu(self):
        self.port_menu.delete(0, 999)

        ports = comports()
        checked = IntVar().set(1)

        if ports:
            for path, _, __ in ports:
                if self.port and path == self.port.port:
                    self.port_menu.add_checkbutton(label=path, command=self.disconnect,
                                                   variable=checked, onvalue=1, offvalue=0)
                else:
                    self.port_menu.add_checkbutton(label=path, command=lambda pathspec=path: self.connect_to(pathspec))

        self.port_menu.add_command(label='No serial ports available', state=DISABLED)

    # def handle_connection_button(self):
    def connect_to(self, portspec):
        self.disconnect()

        # elif self.serial_port_picker.curselection():
            # portspec = self.serial_port_picker.get(self.serial_port_picker.curselection()[0])

        try:
            self.port = Serial(port=portspec, baudrate=115200, timeout=0.2)

            if not self.port.isOpen():
                self.port.open()
            # self.serial_connect_button.configure(text='Disconnect')
            self.status_line.configure(text='Waiting for data...', bg='DarkGoldenrod2')
            self.start_receiving()
        except SerialException:
            logging.exception('dafuq')
            messagebox.showinfo("Can't open port", 'Port already opened by another process')
            self.status_line.configure(text="Can't connect to {0}", bg='firebrick1')
            self.port = None

    def disconnect(self):
        if self.port:
            logging.info('Now disconnecting')

            if self.receiving:
                self.receiving = False
                if self.serial_sniffing_thread and self.serial_sniffing_thread.is_alive():
                    logging.info('Quitting receive')
                    self.serial_sniffing_thread.join(timeout=1)
                    logging.info('No longer receiving')

            if self.port.isOpen():
                logging.info('Port closed')
                self.port.close()
                self.port = None
                # self.serial_connect_button.configure(text='Connect')

        self.status_line.configure(text='Not connected', bg='light goldenrod')

    def start_receiving(self):
        if self.receiving:
            logging.warning('Already receiving, not doing it')
            return

        self.receiving = True
        logging.info('Now receiving')

        self.serial_sniffing_thread = Thread(target=self.handle_packets)
        self.serial_sniffing_thread.isDaemon()
        self.serial_sniffing_thread.start()

    def handle_packets(self):
        while self.receiving and self.port and self.port.isOpen():
            incoming = self.port.readline().decode()
            if incoming:
                logging.debug('Received packet {}'.format(incoming))

                # Packet format:
                # >[./|],[./|],[./|],[./|],[float x],[float y],[float z],[float w],[us since last sample]

                if incoming.count('>') is not 1 or incoming.count(',') is not 8:
                    logging.debug('Packet corrupt - missing delimeter(s)')
                    continue

                # Strip crap that arrived before the delimeter, and also the delimiter itself
                incoming = incoming[incoming.index('>') + 1:].rstrip()

                tokens = incoming.split(',')
                logging.debug(tokens)

                fingers = list(map(lambda i: i is '.', tokens[:4]))
                logging.debug(fingers)

                x, y, z, w = map(float, tokens[4:-1])
                orientation = Quaternion(w, x, y, z)
                delta = float(tokens[-1])

                if delta == 0:
                    frequency = 0
                else:
                    frequency = 1 / (delta / 1000000)

                self.queue.put({'type': 'rx', 'fingers': fingers, 'quat': [x, y, z, w], 'freq': frequency})
                # root.after_idle(self.update_status, fingers, x, y, z, w, frequency)

                logging.debug('Sample parsed')

                self.handle_sample(fingers, orientation, delta)

    def handle_sample(self, fingers, orientation, delta):
        pass

    def update_status(self, fingers, x, y, z, w, frequency):
        def finger(value):
            return '.' if value else '|'

        self.status_line.configure(text='Received {}{}{}{}_, ({},{},{},{}) at {}Hz'.
                                   format(finger(fingers[0]), finger(fingers[1]),
                                          finger(fingers[2]), finger(fingers[3]),
                                          w, x, y, z, round(frequency)),
                                   bg='OliveDrab1')

    def test_event(self):
        print("Sup dawg")
        exit()


def _main():
    app = SomaticTrainerHomeWindow(root)
    app.start()
    root.mainloop()


if __name__ == "__main__":
    _main()
