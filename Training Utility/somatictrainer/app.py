from tkinter import *
from tkinter import messagebox, filedialog, ttk
import serial
from array import array
from serial import Serial, SerialException
from serial.tools.list_ports import comports
import logging
import numpy as np
import quaternion
import threading
from queue import Queue, Empty
from collections import deque
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

        self.last_orientation_received = None
        self.angular_velocity_window = deque(maxlen=5)
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

        self.visualize([e[0] for e in quat_test_data])

        self.file_name_label = Label(right_column, text='No training file open', bg='white', relief=SUNKEN)
        self.file_name_label.pack(fill=X, anchor=N + E)

        self.glyph_picker = ttk.Treeview(right_column, column='count')
        self.glyph_picker.column("#0", width=100, stretch=False)
        self.glyph_picker.column('count', stretch=True)
        self.glyph_picker.heading('count', text='Count')
        self.glyph_picker.pack(fill=BOTH, expand=1, anchor=S + E)
        for glyph in GestureTrainingSet.big_ole_list_o_glyphs:
            self.glyph_picker.insert('', 'end', text="Glyph '{}'".format(glyph), value='0', iid=glyph)
        # first_item = self.glyph_picker.get_children()[0]
        # self.glyph_picker.focus(first_item)
        # self.glyph_picker.selection_set(first_item)

        picker_frame = Frame(right_column, bd=2, relief=SUNKEN)
        picker_frame.grid_rowconfigure(0, weight=1)
        picker_frame.grid_columnconfigure(0, weight=1)

        self.thumbnail_picker = Canvas(picker_frame, bd=0,
                                       width=250, height=150, bg='white')
        self.thumbnail_picker.configure(scrollregion=(0, 0, 250, 2500))
        self.thumbnail_picker.grid(row=0, column=0, sticky=N+S+E+W)

        scrollbar = Scrollbar(picker_frame)
        scrollbar.grid(row=0, column=1, sticky=N+S)

        self.thumbnail_picker.configure(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.thumbnail_picker.yview)

        self.thumbnail_frame = Frame(self.thumbnail_picker)

        self.thumbnail_window = self.thumbnail_picker.create_window((0, 0), window=self.thumbnail_frame, anchor=NW)

        for i in range(10):
            for j in range(10):
                self.thumbnail_frame.grid_columnconfigure(j, weight=1)
                button = Button(self.thumbnail_frame)
                button.grid(row=i, column=j, padx=5, pady=5)

        picker_frame.pack(fill=X, anchor=S+E)

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

                if self.state is self.State.recording:
                    roll, yaw, pitch = quaternion.as_rotation_vector(orientation)

                    x_coord = np.tan(pitch) * 125 + 125
                    y_coord = np.tan(yaw) * -125 + 125

                    logging.info('Rendering ({}, {})'.format(x_coord, y_coord))

                    if self.last_coordinate_visualized:
                        logging.info(self.last_coordinate_visualized)

                        self.path_display.create_line(self.last_coordinate_visualized[0],
                                                             self.last_coordinate_visualized[1],
                                                             x_coord, y_coord,
                                                             width=2, fill='blue')

                    self.last_coordinate_visualized = [x_coord, y_coord]

                    self.path_display.create_oval((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2),
                                                  fill='SeaGreen1')

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

                logging.debug('Sample parsed')

                self.handle_sample(fingers, orientation, microseconds)

    def handle_sample(self, fingers, orientation, microseconds):
        """

        :type fingers: list of bool
        :type orientation: np.quaternion
        :type microseconds: float
        """
        if microseconds == 0:
            frequency = 0
        else:
            frequency = 1 / (microseconds / 1000000)

        if self.state is self.State.recording:
            if self.gesture_anchor:
                orientation = (self.gesture_anchor * orientation)

        if self.last_orientation_received:
            delta = self.last_orientation_received * orientation.inverse()
            norm = np.sqrt(sum((np.square(x) for x in delta.vec)))
            theta = np.arcsin(norm) * 2

            self.angular_velocity_window.append(theta)

            # in_degrees = theta * 180 / np.pi
            # logging.info('Angular velocity {} deg/frame'.format(np.round(in_degrees, 2)))

        self.last_orientation_received = orientation

        gesture_eligible = (len([x for x in self.angular_velocity_window if x > 0.05])
                            and (fingers == self.pointer_gesture or self.state is self.State.recording))
        # gesture_eligible = fingers == self.pointer_gesture

        if gesture_eligible:
            if self.state is not self.State.recording:
                self.path_display.delete(ALL)
                self.last_coordinate_visualized = None

                logging.info('Starting sample!')
                self.status_line.configure(bg='SeaGreen1')
                self.gesture_anchor = orientation.inverse()
                orientation = np.quaternion(1, 0, 0, 0)
                self.gesture_buffer.append((orientation, 0))

                logging.info('Anchor quat: {0}'.format(self.gesture_anchor))
                self.state = self.State.recording
            else:
                self.gesture_buffer.append((orientation, microseconds))
                logging.debug('Relative: {0}'.format(str(orientation)))

        else:
            if self.state is self.State.recording:
                duration = sum([x[1] for x in self.gesture_buffer])

                logging.info('Sample done, {} points, {} ms'.format(len(self.gesture_buffer), duration / 1000))
                self.status_line.configure(bg='OliveDrab1')

                def flash_red():
                    self.path_display.configure(bg='salmon1')
                    self.path_display.after(200, lambda: self.path_display.configure(bg='light grey'))

                def overlay_text(text):
                    overlay = self.path_display.create_text((5, 245), text=text, anchor=SW)

                if len(self.glyph_picker.selection()) is 0:
                    overlay_text('Discarding - no glyph selected')
                    flash_red()
                else:
                    if duration > 300000:
                        ahrs_data = [quaternion.as_rotation_vector(sample[0]) for sample in self.gesture_buffer]
                        min_yaw = min(ahrs[1] for ahrs in ahrs_data)
                        max_yaw = max(ahrs[1] for ahrs in ahrs_data)
                        min_pitch = min(ahrs[2] for ahrs in ahrs_data)
                        max_pitch = max(ahrs[2] for ahrs in ahrs_data)

                        selected_glyph = self.glyph_picker.selection()[0]

                        if selected_glyph not in GestureTrainingSet.short_glyphs \
                                and max_yaw - min_yaw > np.pi / 8 or max_pitch - min_pitch > np.pi / 8:

                            self.training_set.add(Gesture(selected_glyph, list(self.gesture_buffer)))

                            self.visualize([x[0] for x in self.gesture_buffer])

                            overlay_text('Accepted sample #{} for glyph {}'.format(
                                self.training_set.count(selected_glyph), selected_glyph))

                            self.path_display.configure(bg='pale green')
                            self.path_display.after(200, lambda: self.path_display.configure(bg='light grey'))

                        else:
                            overlay_text('Discarding - too small')
                            flash_red()
                    else:
                        overlay_text('Discarding - too short')
                        flash_red()

                self.cancel_gesture()
                self.state = self.State.connected

        if self.queue.empty():
            self.queue.put({'type': 'rx',
                            'fingers': fingers,
                            'quat': orientation,
                            'freq': frequency})

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
        self.last_orientation_received = None
        self.last_coordinate_visualized = None

    def test_event(self):
        print("Sup dawg")
        exit()

    def visualize(self, path):
        self.path_display.delete(ALL)

        dots = []

        for quat in path:
            roll, yaw, pitch = quaternion.as_rotation_vector(quat)

            x_coord = np.tan(pitch) * 120 + 122
            y_coord = np.tan(yaw) * -120 + 122
            dots.append([x_coord, y_coord])

        min_x = min([coord[0] for coord in dots])
        max_x = max([coord[0] for coord in dots])
        min_y = min([coord[1] for coord in dots])
        max_y = max([coord[1] for coord in dots])

        last_point = None

        for x, y in dots:
            x_coord = _scale(x, min_x, max_x, 5, 240)
            y_coord = _scale(y, min_y, max_y, 5, 240)

            logging.debug('Plotting ({}, {}) - scaled to ({}, {})'.format(x, y, x_coord, y_coord))

            if last_point:
                self.path_display.create_line(last_point[0],
                                                     last_point[1],
                                                     x_coord, y_coord,
                                                     width=2, fill='blue')

            last_point = [x_coord, y_coord]

            self.path_display.create_oval((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2), fill='SeaGreen1')


def _scale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

quat_test_data = [(np.quaternion(1, 0, 0, 0), 0), (np.quaternion(0.993842486840799, -0.0122157115900289, -0.000297944185122676, -0.00337670076472341), 20654.0), (np.quaternion(1.00009931472837, -0.0136061177872679, -0.00506505114708515, -0.00933558446717647), 20654.0), (np.quaternion(0.994835634124541, -0.0156917270831264, -0.0238355348098124, -0.0493594200019863), 20653.0), (np.quaternion(0.993743172112424, -0.0185718542059788, -0.0240341642665607, -0.0663422385539775), 20655.0), (np.quaternion(0.992948654285431, -0.0180752805641077, -0.0303903068825107, -0.0955407686959976), 20654.0), (np.quaternion(0.987684973681597, -0.0201608898599663, -0.0491607905452378, -0.135564604230807), 20653.0), (np.quaternion(0.977256927202304, -0.0179759658357335, -0.0629655377892541, -0.152646737511173), 20655.0), (np.quaternion(0.97904459231304, -0.0219485549707022, -0.0801469857979939, -0.169430926606416), 20653.0), (np.quaternion(0.977058297745556, -0.0153937828980038, -0.0983215810904758, -0.158506306485252), 20654.0), (np.quaternion(0.979839110140034, -0.00288012712285235, -0.110239348495382, -0.148972092561327), 20655.0), (np.quaternion(0.981527460522395, 0.00208560929585858, -0.139040619723905, -0.108650312841394), 20653.0), (np.quaternion(0.980434998510279, -0.00546231006058201, -0.155924123547522, -0.0778627470453868), 20654.0), (np.quaternion(0.984109643460125, -0.0070513457145695, -0.166550799483563, -0.0484655874466183), 20655.0), (np.quaternion(0.974476114807826, -0.010030787565796, -0.190684278478498, 0.0114211937630351), 20654.0), (np.quaternion(0.976661038832059, -0.00427053332009136, -0.190287019565001, 0.0453868308670176), 20653.0), (np.quaternion(0.97715761247393, -0.00814380772668588, -0.17777336378985, 0.0868010725990664), 20655.0), (np.quaternion(0.981428145794021, -0.0225444433409475, -0.164266560730956, 0.100506505114709), 20653.0), (np.quaternion(0.978349389214421, -0.0384347998808224, -0.146191280166849, 0.103188002780812), 20653.0), (np.quaternion(0.987784288409971, -0.0766709703048963, -0.101201708213328, 0.102691429138941), 20655.0), (np.quaternion(0.989472638792333, -0.086602443142318, -0.0768695997616447, 0.103982520607806), 20653.0), (np.quaternion(0.983811699275002, -0.0980236369053531, -0.0479690138047473, 0.0942496772271328), 20654.0), (np.quaternion(0.995928096136657, -0.105472241533419, 0.006852716257821, 0.0648525176283643), 20655.0), (np.quaternion(0.990565100804449, -0.113516734531731, 0.029595789055517, 0.0429039626576622), 20653.0), (np.quaternion(0.991160989174695, -0.121660542258417, 0.0704141424173205, -0.0205581487734631), 20655.0), (np.quaternion(0.986393882212732, -0.122951633727282, 0.0808421888966133, -0.0669381269242229), 20654.0), (np.quaternion(0.983017181448009, -0.117985897308571, 0.0853113516734531, -0.10855099811302), 20654.0), (np.quaternion(0.969510378389115, -0.107458536100904, 0.0698182540470751, -0.179461714172212), 20655.0), (np.quaternion(0.964147383056907, -0.0959380276094945, 0.0561128215314331, -0.210149965239845), 20653.0), (np.quaternion(0.964743271427153, -0.0891846260800477, 0.0437977952130301, -0.234581388419903), 20654.0), (np.quaternion(0.967623398550005, -0.0707120866024432, -0.00963352865229915, -0.243122455060085), 20655.0), (np.quaternion(0.96255834740292, -0.0604826695798987, -0.0461813486940114, -0.238256033369749), 20653.0), (np.quaternion(0.969907637302612, -0.0440957393981528, -0.103883205879432, -0.188201410269143), 20653.0), (np.quaternion(0.977852815572549, -0.0405204091766809, -0.137451584069918, -0.153838514251664), 20655.0), (np.quaternion(0.973284338067335, -0.034164266560731, -0.161485748336478, -0.107557850829278), 20653.0), (np.quaternion(0.978150759757672, -0.0511470851127222, -0.193663720329725, -0.0227430727976959), 20653.0), (np.quaternion(0.975568576819942, -0.0513457145694707, -0.199523289303804, 0.0126129705035257), 20654.0), (np.quaternion(0.971595987684974, -0.0578011719137949, -0.19942397457543, 0.0432019068427848), 20653.0), (np.quaternion(0.97527063263482, -0.0789552090575033, -0.173602145198133, 0.0813387625384845), 20654.0), (np.quaternion(0.978051445029298, -0.0860065547720728, -0.149071407289701, 0.0996126725593406), 20655.0), (np.quaternion(0.978647333399543, -0.118383156222068, -0.0884894229814282, 0.0926606415731453), 20653.0), (np.quaternion(0.983712384546628, -0.128612573244612, -0.051941602939716, 0.0877942198828086), 20653.0), (np.quaternion(0.981328831065647, -0.126725593405502, -0.00576025424570462, 0.0812394478101102), 20655.0), (np.quaternion(0.984904161287119, -0.124937928294766, 0.0584963750124144, 0.0331711192769888), 20653.0), (np.quaternion(0.978349389214421, -0.122256430628662, 0.0861058695004469, 0.00357533022147188), 20653.0), (np.quaternion(0.984308272916874, -0.122355745357037, 0.104181150064555, -0.0379382262389512), 20655.0), (np.quaternion(0.971099414043103, -0.128016684874367, 0.118979044592313, -0.11232495779124), 20653.0), (np.quaternion(0.962955606316417, -0.129009832158109, 0.117191379481577, -0.152547422782799), 20653.0), (np.quaternion(0.955606316416725, -0.106266759360413, 0.101996226040322, -0.220081438077267), 20655.0), (np.quaternion(0.95620220478697, -0.0995133578309664, 0.0896811997219187, -0.244512861257324), 20654.0), (np.quaternion(0.953421392392492, -0.0924620121163969, 0.065150461813487, -0.262786771278181), 20655.0), (np.quaternion(0.953719336577614, -0.0695203098619526, 0.0225444433409474, -0.283742178965141), 20654.0), (np.quaternion(0.954613169132982, -0.0640579998013706, -0.0126129705035256, -0.272618929387228), 20654.0), (np.quaternion(0.954116595491111, -0.0648525176283644, -0.0418115006455458, -0.266262786771278), 20655.0), (np.quaternion(0.967524083821631, -0.0668388121958487, -0.103188002780812, -0.219286920250273), 20654.0), (np.quaternion(0.967822028006754, -0.0681299036647135, -0.126030390306883, -0.183732247492303), 20653.0), (np.quaternion(0.972986393882213, -0.0817360214519813, -0.164365875459331, -0.11113318105075), 20655.0), (np.quaternion(0.971595987684974, -0.0926606415731454, -0.175091866123746, -0.0681299036647134), 20653.0), (np.quaternion(0.970205581487735, -0.103585261694309, -0.185817856788162, -0.0251266262786771), 20653.0), (np.quaternion(0.976065150461814, -0.118979044592313, -0.159598768497368, 0.0469758665210051), 20655.0), (np.quaternion(0.973979541165955, -0.118383156222068, -0.136259807329427, 0.0759757672062767), 20653.0), (np.quaternion(0.981527460522395, -0.119475618234184, -0.105472241533419, 0.0928592710298937), 20653.0), (np.quaternion(0.986493196941106, -0.135663918959182, -0.0274108650312841, 0.106068129903665), 20655.0), (np.quaternion(0.985599364385738, -0.141126229019764, 0.00774654881318902, 0.0949448803257523), 20653.0), (np.quaternion(0.984109643460125, -0.148674148376204, 0.0719038633429338, 0.0604826695798987), 20653.0), (np.quaternion(0.977554871387427, -0.1459926507101, 0.0995133578309663, 0.0308868805243818), 20655.0), (np.quaternion(0.972489820240342, -0.150660442943689, 0.116098917469461, -0.00327738603634914), 20653.0), (np.quaternion(0.970006952030986, -0.155129605720528, 0.138544046082034, -0.0727976958983017), 20653.0), (np.quaternion(0.966630251266263, -0.150163869301817, 0.143013208858874, -0.114410567087099), 20656.0), (np.quaternion(0.963253550501539, -0.145198132883107, 0.147482371635714, -0.156023438275896), 20654.0), (np.quaternion(0.95739398152746, -0.129804349985103, 0.12126328334492, -0.228125931075578), 20653.0), (np.quaternion(0.951832356738504, -0.135266660045685, 0.108650312841394, -0.25593405502036), 20656.0), (np.quaternion(0.938921442049856, -0.122653689542159, 0.0641573145297447, -0.30350580991161), 20654.0), (np.quaternion(0.942099513357831, -0.120369450789552, 0.041016982818552, -0.31552289204489), 20653.0), (np.quaternion(0.935644056013507, -0.116396861654583, 0.0104280464792928, -0.315423577316516), 20656.0), (np.quaternion(0.941106366074089, -0.102393484953819, -0.0538285827788261, -0.311550302909922), 20653.0), (np.quaternion(0.940609792432218, -0.103188002780812, -0.0830271129208462, -0.305194160293972), 20656.0), (np.quaternion(0.941205680802463, -0.101102393484954, -0.112027013606118, -0.28185519912603), 20653.0), (np.quaternion(0.944185122653689, -0.120468765517926, -0.150759757672063, -0.24322176978846), 20654.0), (np.quaternion(0.949349488529149, -0.129407091071606, -0.172410368457642, -0.218393087694905), 20655.0), (np.quaternion(0.94795908233191, -0.14033171119277, -0.183136359122058, -0.175389810308869), 20653.0), (np.quaternion(0.955010428046479, -0.14221869103188, -0.18154732346807, -0.152150163869302), 20653.0), (np.quaternion(0.96067136756381, -0.150362498758566, -0.17399940411163, -0.133677624391697), 20655.0), (np.quaternion(0.964246697785281, -0.157910418115006, -0.143112523587248, -0.0862051842288211), 20653.0), (np.quaternion(0.979441851226537, -0.154136458436786, -0.123050948455656, -0.0705134571456947), 20653.0), (np.quaternion(0.981030886880524, -0.15046181348694, -0.0936537888568875, -0.0598867812096534), 20655.0), (np.quaternion(0.984407587645248, -0.126129705035257, -0.0359519316714669, -0.0415135564604231), 20654.0), (np.quaternion(0.992452080643559, -0.106862647730659, -0.0124143410467773, -0.0397258913496872), 20653.0), (np.quaternion(0.985003476015493, -0.0947462508690039, -0.00278081239447813, -0.0414142417320489), 20655.0), (np.quaternion(0.993147283742179, -0.0890853113516735, 0.0156917270831264, -0.0489621610884894), 20652.0), (np.quaternion(0.99175687754494, -0.0953421392392492, 0.0216506107855795, -0.0537292680504519), 20654.0), (np.quaternion(0.991657562816566, -0.0817360214519814, 0.0267156619326646, -0.0443936835832754), 20655.0), (np.quaternion(0.996424669778528, -0.0757771377495283, 0.0329724898202403, -0.0457840897805145), 20653.0), (np.quaternion(0.990167841890953, -0.0743867315522892, 0.0377395967822028, -0.0398252060780614), 20654.0), (np.quaternion(0.994934948852915, -0.0684278478498362, 0.0439964246697785, -0.0412156122753005), 20655.0), (np.quaternion(0.988678120965339, -0.0670374416525972, 0.048763531631741, -0.0352567285728474), 20652.0)]
