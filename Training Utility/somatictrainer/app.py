from tkinter import *
from tkinter import messagebox, filedialog, ttk
import serial
from serial import Serial, SerialException
from serial.tools.list_ports import comports
import logging
import numpy as np
import quaternion
import threading
from queue import Queue, Empty
import json
import os
from PIL import Image, ImageTk
from enum import Enum
from somatictrainer.gestures import Gesture, GestureTrainingSet

logging.basicConfig(level=logging.INFO)


class SomaticTrainerHomeWindow(Frame):
    port: serial.Serial
    serial_sniffing_thread: threading.Thread
    training_set: GestureTrainingSet

    # pointer_gesture = [False, False, False, True]
    pointer_gesture = [True, True, True, False]

    class State(Enum):
        quitting = -1
        disconnected = 0
        connecting = 1
        connected = 2
        recording = 3
        processing = 4

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        hand_icon_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Hands')

        unknown_hand_icon_bitmap = Image.open(os.path.join(hand_icon_directory, 'Unknown.png'))
        unknown_hand_icon_bitmap.thumbnail((250, 250))
        self.unknown_hand_icon = ImageTk.PhotoImage(image=unknown_hand_icon_bitmap)

        hand_bitmaps = []
        self.hand_icons = {}

        for i in range(16):
            hand_bitmap = Image.open(os.path.join(hand_icon_directory, '{:04b}.png'.format(i)))
            hand_bitmap.thumbnail((250, 250))

            hand_bitmaps.append(hand_bitmap)
            self.hand_icons[i] = ImageTk.PhotoImage(image=hand_bitmap)

        self.state = self.State.disconnected

        self.queue = Queue()
        self.port = None
        self.serial_sniffing_thread = None
        self.receiving = False
        self.last_hand_id = -1

        self.open_file_pathspec = ''
        self.open_file_has_been_modified = False
        self.training_set = GestureTrainingSet()

        self.gesture_buffer = []
        self.gesture_anchor = None

        self.last_coordinate_visualized = None

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
        self.hand_display.create_image((0, 0), image=self.unknown_hand_icon, anchor=N + W)
        self.hand_display.pack(fill=X)

        Label(left_column, text='Path').pack(fill=X)
        self.path_display = Canvas(left_column, width=250, height=250)
        self.path_display.pack(fill=X)

        self.file_name_label = Label(right_column, text='No training file open', bg='white', relief=SUNKEN)
        self.file_name_label.pack(fill=X, anchor=N + E)

        self.glyph_picker = ttk.Treeview(right_column, column='count')
        self.glyph_picker.column("#0", width=100, stretch=False)
        self.glyph_picker.column('count', stretch=True)
        self.glyph_picker.heading('count', text='Count')
        self.glyph_picker.pack(fill=BOTH, expand=1, anchor=S + E)
        for glyph in GestureTrainingSet.big_ole_list_o_glyphs:
            self.glyph_picker.insert('', 'end', text="Glyph '{}'".format(glyph), value='0')
        first_item = self.glyph_picker.get_children()[0]
        self.glyph_picker.focus(first_item)
        self.glyph_picker.selection_set(first_item)

        self.status_line = Label(self, text='Not connected', bd=1, relief=SUNKEN, anchor=W, bg='light goldenrod')

        left_column.grid(row=0, column=0, sticky=N)
        right_column.grid(row=0, column=1, sticky=N + S + E + W)
        self.status_line.grid(row=1, column=0, columnspan=2, sticky=S + E + W)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def start(self):
        # self.reload_serial_port_picker()
        self.master.after(10, self.queue_handler)
        self.connect_to('COM23')

    def stop(self):
        if self.open_file_has_been_modified:
            response = messagebox.askyesnocancel('Unsaved changes',
                                                 'You have unsaved changes. Are you sure you want to quit?')
            if response:
                self.save_file()
            if response is None:
                return

        self.state = self.State.quitting
        self.receiving = False
        self.queue.put({'type': 'quit'})

    def queue_handler(self):
        try:
            command = self.queue.get(block=False)

            if command['type'] is 'ack':
                if self.state is self.State.connecting:
                    self.state = self.State.connected
                    logging.info('Got ack - connected')

            if command['type'] is 'rx':
                orientation = command['quat']
                if self.state is not self.State.disconnected and self.state is not self.State.quitting:
                    self.update_status(command['fingers'],
                                       orientation.w, orientation.x, orientation.y, orientation.z,
                                       command['freq'])

                    fingers = command['fingers']
                    hand_id = fingers[0] * 0b1000 + fingers[1] * 0b0100 + fingers[2] * 0b0010 + fingers[3] * 0b0001
                    if hand_id != self.last_hand_id:
                        self.hand_display.create_image((0, 0), image=self.hand_icons[hand_id], anchor=N + W)
                        self.last_hand_id = hand_id

                if self.state is self.State.recording or self.state is self.State.connected:
                    roll, yaw, pitch = quaternion.as_rotation_vector(orientation)

                    x_coord = np.tan(yaw) * 125 + 125
                    y_coord = np.tan(pitch) * 125 + 125

                    logging.debug('yaw {0} pitch {1} roll {2}'.format(yaw * 180 / np.pi,
                                                                      pitch * 180 / np.pi,
                                                                      roll * 180 / np.pi))

                    logging.debug('x {0} y {1}'.format(x_coord, y_coord))

                    if self.last_coordinate_visualized:
                        line = self.path_display.create_line(self.last_coordinate_visualized[0],
                                                             self.last_coordinate_visualized[1],
                                                             x_coord, y_coord)

                        self.path_display.after(1000, lambda x=line: self.path_display.delete(x))

                    self.last_coordinate_visualized = [x_coord, y_coord]

                    dot = self.path_display.create_oval((x_coord, y_coord, x_coord + 3, y_coord + 3), fill='blue')

                    self.path_display.after(1000, lambda x=dot: self.path_display.delete(x))

            if command['type'] is 'quit':
                self.master.destroy()
                return

        except Empty:
            pass

        self.master.after(10, self.queue_handler)

    def new_file(self):
        if self.state is self.State.recording:
            self.state = self.State.connected
            self.cancel_gesture()

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
        if self.state is self.State.recording:
            self.state = self.State.connected
            self.cancel_gesture()

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
        if self.state is self.State.recording:
            self.state = self.State.connected
            self.cancel_gesture()

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

    def populate_serial_port_menu(self):
        self.port_menu.delete(0, 999)

        ports = comports()
        checked = IntVar().set(1)
        not_checked = IntVar().set(0)

        if ports:
            for path, _, __ in ports:
                if self.port and path == self.port.port:
                    self.port_menu.add_checkbutton(label=path, command=self.disconnect,
                                                   variable=checked, onvalue=1, offvalue=0)
                else:
                    self.port_menu.add_checkbutton(label=path, command=lambda pathspec=path: self.connect_to(pathspec),
                                                   variable=not_checked, onvalue=1, offvalue=0)

        else:
            self.port_menu.add_command(label='No serial ports available', state=DISABLED)

    # def handle_connection_button(self):
    def connect_to(self, portspec):
        self.disconnect()

        try:
            self.port = Serial(port=portspec, baudrate=115200, timeout=0.2)

            if not self.port.isOpen():
                self.port.open()
            # self.serial_connect_button.configure(text='Disconnect')
            self.state = self.State.connecting
            self.status_line.configure(text='Waiting for response...', bg='DarkGoldenrod2')
            self.port.write(bytes('>AT\r\n'.encode('utf-8')))
            self.start_receiving()
        except SerialException:
            logging.exception('dafuq')
            messagebox.showinfo("Can't open port", 'Port already opened by another process')
            self.state = self.State.disconnected
            self.status_line.configure(text="Can't connect to {0}", bg='firebrick1')
            self.port = None

    def disconnect(self):
        if self.port:
            logging.info('Now disconnecting')

            if self.receiving:
                self.receiving = False
                if threading.current_thread() is not self.serial_sniffing_thread \
                        and self.serial_sniffing_thread \
                        and self.serial_sniffing_thread.is_alive():
                    logging.info('Quitting receive')
                    self.serial_sniffing_thread.join(timeout=1)
                    logging.info('No longer receiving')

            if self.port.isOpen():
                logging.info('Port closed')
                self.port.close()
                self.port = None
                # self.serial_connect_button.configure(text='Connect')

        self.last_hand_id = -1
        self.cancel_gesture()
        self.hand_display.create_image((0, 0), image=self.unknown_hand_icon, anchor=N + W)
        self.state = self.State.disconnected
        self.status_line.configure(text='Not connected', bg='light goldenrod')

    def start_receiving(self):
        if self.receiving:
            logging.warning('Already receiving, not doing it')
            return

        self.receiving = True
        logging.info('Now receiving')

        self.serial_sniffing_thread = threading.Thread(target=self.handle_packets)
        self.serial_sniffing_thread.isDaemon()
        self.serial_sniffing_thread.start()

    def handle_packets(self):
        while self.receiving and self.port and self.port.isOpen():
            try:
                incoming = self.port.readline().decode()
            except SerialException:
                logging.exception('Lost serial connection, bailing out')
                self.disconnect()
                return

            if incoming:
                logging.debug('Received packet {}'.format(incoming))

                # Packet format:
                # >[./|],[./|],[./|],[./|],[float x],[float y],[float z],[float w],[us since last sample]

                if incoming.count('>') is not 1:
                    logging.debug('Packet corrupt - missing delimeter(s)')
                    continue

                # Strip crap that arrived before the delimeter, and also the delimiter itself
                incoming = incoming[incoming.index('>') + 1:].rstrip()

                if incoming == 'OK':
                    logging.info('Received ack')
                    self.queue.put({'type': 'ack'})
                    continue

                if incoming.count(',') is not 8:
                    logging.debug('Packet corrupt - insufficient fields')
                    continue

                tokens = incoming.split(',')
                logging.debug('Tokens: {0}'.format(tokens))

                fingers = list(map(lambda i: i is '.', tokens[:4]))
                logging.debug('Fingers: {0}'.format(fingers))

                x, y, z, w = map(float, tokens[4:-1])
                orientation = np.quaternion(w, x, y, z)
                microseconds = float(tokens[-1])

                if microseconds == 0:
                    frequency = 0
                else:
                    frequency = 1 / (microseconds / 1000000)

                if self.queue.empty():
                    self.queue.put({'type': 'rx',
                                    'fingers': fingers,
                                    'quat': self.gesture_anchor * orientation
                                    if self.state is self.State.recording and self.gesture_anchor
                                    else orientation,
                                    'freq': frequency})
                # root.after_idle(self.update_status, fingers, x, y, z, w, frequency)

                logging.debug('Sample parsed')

                self.handle_sample(fingers, orientation, microseconds)

    def handle_sample(self, fingers, orientation, microseconds):
        """

        :type fingers: list of bool
        :type orientation: np.quaternion
        :type microseconds: float
        """
        if fingers == self.pointer_gesture:
            if self.state is not self.State.recording:
                logging.info('Starting sample!')
                self.status_line.configure(bg='SeaGreen1')
                self.gesture_anchor = orientation.inverse()
                self.gesture_buffer.append((orientation * self.gesture_anchor, 0))

                logging.info('Start quat: {0} -- Anchor quat: {1}'.format(orientation, self.gesture_anchor))
                self.state = self.State.recording
            else:
                relative_orientation = self.gesture_anchor * orientation
                self.gesture_buffer.append((relative_orientation, microseconds))
                logging.debug('Relative: {0}'.format(str(relative_orientation)))

        else:
            if self.state is self.State.recording:
                logging.info('Sample done, {} points'.format(len(self.gesture_buffer)))
                self.status_line.configure(bg='OliveDrab1')

                if len(self.gesture_buffer) > 1:
                    self.training_set.add(Gesture(self.glyph_picker.selection()[0], list(self.gesture_buffer)))

                self.cancel_gesture()
                self.state = self.State.connected

    def update_status(self, fingers, w, x, y, z, frequency):
        def finger(value):
            return '.' if value else '|'

        self.status_line.configure(text='Received {0}{1}{2}{3}_, ({4:.2f},{5:.2f},{6:.2f},{7:.2f}) at {8}Hz'.
                                   format(finger(fingers[0]), finger(fingers[1]),
                                          finger(fingers[2]), finger(fingers[3]),
                                          w, x, y, z, round(frequency)),
                                   bg='SeaGreen1' if self.state is self.State.recording else 'OliveDrab1')

    def cancel_gesture(self):
        del self.gesture_buffer[:]
        self.gesture_anchor = None
        self.path_display.delete(ALL)

    def test_event(self):
        print("Sup dawg")
        exit()


def _scale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
