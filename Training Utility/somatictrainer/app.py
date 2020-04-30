from tkinter import *
from tkinter import messagebox, filedialog, ttk, font

import serial
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
from PIL import Image, ImageTk, ImageDraw
from enum import Enum
import tensorflow.keras as keras
from somatictrainer.util import *
from somatictrainer.gestures import Gesture, GestureTrainingSet, standard_gesture_time, sampling_rate

logging.basicConfig(level=logging.INFO)


class SomaticTrainerHomeWindow(Frame):
    model: keras.Model
    port: serial.Serial
    serial_sniffing_thread: threading.Thread
    training_set: GestureTrainingSet

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

        self.example_thumbnails = {}

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

        self.bearing_zero = None
        self.gesture_hand_position = None
        self.gesture_hand_velocity = None
        self.last_bearing_received = None
        self.angular_velocity_window = deque(maxlen=5)
        self.starting_velocity_estimation_buffer = deque(maxlen=10)
        self.last_coordinate_visualized = None

        self.master.title('Somatic Trainer')
        self.pack(fill=BOTH, expand=1)

        self.menu_bar = Menu(master)

        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label='New', command=self.new_file)
        self.file_menu.add_command(label='Open', command=self.open_file)
        self.save_entry_index = 2
        self.file_menu.add_command(label='Save', command=self.save_file, state=DISABLED)
        self.save_as_entry_index = 3
        self.file_menu.add_command(label='Save as...', command=self.save_as, state=DISABLED)
        self.menu_bar.add_cascade(label='File', menu=self.file_menu)

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

        label_width_locking_frame = Frame(right_column, height=20, width=250)
        label_width_locking_frame.pack_propagate(0)  # This prevents a long filename from stretching the window
        label_width_locking_frame.pack(fill=X, anchor=N + E)
        self.file_name_label = Label(label_width_locking_frame, text='No training file open', bg='white', relief=SUNKEN)
        self.file_name_label.pack(fill=BOTH, expand=1)

        self.glyph_picker = ttk.Treeview(right_column, column='count')
        self.glyph_picker.column("#0", width=100, stretch=False)
        self.glyph_picker.column('count', stretch=True)
        self.glyph_picker.heading('count', text='Count')
        self.glyph_picker.pack(fill=BOTH, expand=1, anchor=S + E)
        # first_item = self.glyph_picker.get_children()[0]
        # self.glyph_picker.focus(first_item)
        # self.glyph_picker.selection_set(first_item)
        self.reload_glyph_picker()
        self.glyph_picker.bind('<<TreeviewSelect>>', lambda x: self.reload_example_list())

        picker_frame = Frame(right_column, bd=2, relief=SUNKEN)
        picker_frame.grid_rowconfigure(0, weight=1)
        picker_frame.grid_columnconfigure(0, weight=1)

        self.thumbnail_canvas = Canvas(picker_frame, bd=0, width=250, height=150, bg='white')
        self.thumbnail_canvas.grid(row=0, column=0, sticky=N + S + E + W)

        scrollbar = Scrollbar(picker_frame)
        scrollbar.grid(row=0, column=1, sticky=N + S)

        self.thumbnail_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.thumbnail_canvas.yview)

        def bind_wheel_to_thumbnails(event):
            self.thumbnail_canvas.bind_all('<MouseWheel>', on_wheel_scroll)

        def unbind_wheel_from_thumbnails(event):
            self.thumbnail_canvas.unbind('<MouseWheel>')

        def on_wheel_scroll(event):
            self.thumbnail_canvas.yview_scroll(int(event.delta / -120), 'units')

        self.thumbnail_frame = Frame(self.thumbnail_canvas)
        self.thumbnail_frame.bind('<Enter>', bind_wheel_to_thumbnails)
        self.thumbnail_frame.bind('<Leave>', unbind_wheel_from_thumbnails)

        self.thumbnail_canvas.create_window((0, 0), window=self.thumbnail_frame, anchor=NW)

        for i in range(5):
            self.thumbnail_frame.grid_columnconfigure(i, weight=1)

        picker_frame.pack(fill=X, anchor=S + E)

        self.status_line = Label(self, text='Not connected', bd=1, relief=SUNKEN, anchor=W, bg='light goldenrod')

        left_column.grid(row=0, column=0, sticky=N)
        right_column.grid(row=0, column=1, sticky=N + S + E + W)
        self.status_line.grid(row=1, column=0, columnspan=2, sticky=S + E + W)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Debug code!
        self.model = keras.models.load_model(
            'E:\Dropbox\Projects\Source-Controlled Projects\Somatic\Training Utility\\2020-52-20T00-52.h5')

    def reload_glyph_picker(self):
        scroll_position = self.glyph_picker.yview()[0]
        selected_item = self.get_selected_glyph()

        self.glyph_picker.delete(*self.glyph_picker.get_children())

        for glyph in GestureTrainingSet.big_ole_list_o_glyphs:
            self.glyph_picker.insert('', 'end', text="Glyph '{}'".format(glyph),
                                     value=str(self.training_set.count(glyph)) if self.training_set else '0',
                                     iid=glyph)

        if selected_item is not None:
            self.glyph_picker.focus(selected_item)
            self.glyph_picker.selection_set(selected_item)
            self.glyph_picker.yview_moveto(scroll_position)

    def get_selected_glyph(self):
        selected_item = self.glyph_picker.selection()[0] if len(self.glyph_picker.selection()) else None
        return selected_item

    def reload_example_list(self):
        buttons = self.thumbnail_frame.winfo_children()
        for button in buttons:
            button.grid_forget()
            button.destroy()

        selected_glyph = self.get_selected_glyph()
        if selected_glyph is None or not self.training_set.count(selected_glyph):
            return

        for index, example in enumerate(self.training_set.get_examples_for(selected_glyph)):
            if example in self.example_thumbnails:
                thumbnail = self.example_thumbnails[example]
            else:
                thumbnail = ImageTk.PhotoImage(
                    image=_gesture_to_image(
                        self.training_set.get_examples_for(selected_glyph)[index].normalized_data, 50, 50, 2, 2, 2))
                self.example_thumbnails[example] = thumbnail

            button = Button(self.thumbnail_frame, image=thumbnail)
            button.image = thumbnail  # Button doesn't have an image field - this monkey patch retains a reference
            button.setvar('example_id', index)
            button.configure(command=
                             lambda i=index, g=selected_glyph:
                             self.visualize(self.training_set.get_examples_for(g)[i].normalized_data))

            def delete_on_right_click(g, i):
                logging.info('Removing glyph {} #{}'.format(g, i))
                self.training_set.remove_at(g, i)
                self.reload_example_list()
                self.reload_glyph_picker()
                self.open_file_has_been_modified = True

            # Right click on OSX
            button.bind('<Button-2>', lambda event, g=selected_glyph, i=index: delete_on_right_click(g, i))

            # Right click on Windows
            button.bind('<Button-3>', lambda event, g=selected_glyph, i=index: delete_on_right_click(g, i))

            button.grid(row=int(index / 5), column=index % 5)

        self.thumbnail_canvas.configure(
            # scrollregion=(0, 0, 250, int(self.training_set.count(selected_glyph) / 5) * 50))
            scrollregion=self.thumbnail_canvas.bbox(ALL))

    def start(self):
        self.master.after(10, self.queue_handler)
        self.connect_to('COM23')

    def stop(self):
        if self.open_file_has_been_modified:
            response = messagebox.askyesnocancel('Unsaved changes',
                                                 'Save before quitting?')
            if response:
                self.save_file()
            elif response is None:
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
                bearing = command['bearing']
                accelerometer = command['accel']
                position = command['position']
                if self.state is not self.State.disconnected and self.state is not self.State.quitting:
                    self.update_status(command['fingers'],
                                       bearing, accelerometer,
                                       command['freq'])

                    fingers = command['fingers']
                    hand_id = fingers[0] * 0b1000 + fingers[1] * 0b0100 + fingers[2] * 0b0010 + fingers[3] * 0b0001
                    if hand_id != self.last_hand_id:
                        self.hand_display.create_image((0, 0), image=self.hand_icons[hand_id], anchor=N + W)
                        self.last_hand_id = hand_id

                if self.state is self.State.recording:
                    x_coord = np.tan(bearing[0]) * 250 + 125/2
                    y_coord = np.tan(bearing[1]) * 250 + 125/2

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

                    if position is not None:
                        x_coord = position[1] + 125
                        y_coord = position[2] + 125

                        self.path_display.create_oval((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2),
                                                      fill='red')

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

        new_file_pathspec = filedialog.asksaveasfilename(
            title='Create training file',
            filetypes=(('JSON dictionary', '*.json'), ('All files', '*')))
        if new_file_pathspec:
            new_file_pathspec += '.json'
            self.training_set = GestureTrainingSet()
            self.open_file_pathspec = new_file_pathspec
            self.file_menu.entryconfigure(self.save_entry_index, state=NORMAL)
            self.file_menu.entryconfigure(self.save_as_entry_index, state=NORMAL)
            self.file_name_label.configure(text=self.open_file_pathspec, anchor=E)
            self.open_file_has_been_modified = True
            self.reload_glyph_picker()

    def open_file(self):
        file_to_open = filedialog.askopenfilename(title='Select training data',
                                                  filetypes=(('JSON dictionary', '*.json'), ('All files', '*')))

        if file_to_open:
            try:
                training_set = GestureTrainingSet.load(file_to_open)
                self.training_set = training_set
                self.reload_glyph_picker()

                self.open_file_pathspec = file_to_open
                self.file_menu.entryconfigure('Save', state=NORMAL)
                self.file_menu.entryconfigure('Save as...', state=NORMAL)
                self.file_name_label.configure(text=self.open_file_pathspec, anchor=E)
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
                self.open_file_pathspec = file_to_create + '.json'
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
                # >[./|],[./|],[./|],[./|],
                # [float o.h],[float o.p],[float o.r],
                # [float a.x], [float a.y], [float a.z], [us since last sample]

                if incoming.count('>') is not 1:
                    logging.debug('Packet corrupt - missing delimeter(s)')
                    continue

                # Strip crap that arrived before the delimeter, and also the delimiter itself
                incoming = incoming[incoming.index('>') + 1:].rstrip()

                if incoming == 'OK':
                    logging.info('Received ack')
                    self.queue.put({'type': 'ack'})
                    continue

                if incoming.count(',') is not 10:
                    logging.debug('Packet corrupt - insufficient fields')
                    continue

                tokens = incoming.split(',')
                logging.debug('Tokens: {0}'.format(tokens))

                fingers = list(map(lambda i: i is '.', tokens[:4]))
                logging.debug('Fingers: {0}'.format(fingers))

                bearing = np.array([float(t) for t in tokens[4:7]])

                # orientation = quaternion.quaternion(w, x, y, z)

                local_acceleration = np.array([float(t) for t in tokens[7:10]])

                microseconds = float(tokens[-1])

                logging.debug('Sample parsed')

                self.handle_sample(fingers, bearing, local_acceleration, microseconds)

    def handle_sample(self, fingers, bearing, local_acceleration, microseconds):
        """

        :type fingers: list of bool
        :type bearing: np.array
        :type local_acceleration: np.array
        :type microseconds: float
        """

        # x_coord = np.tan(orientation.x + np.pi) * 125 + 125
        # y_coord = np.tan(orientation.y) * -125 + 125
        #
        # logging.info('Rendering ({}, {})'.format(x_coord, y_coord))
        #
        # self.path_display.create_oval((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2),
        #                               fill='SeaGreen1')

        # orientation = quaternion.from_euler_angles(bearing[1], bearing[2], bearing[0])
        orientation = custom_euler_to_quat(bearing[0], bearing[1], bearing[2])
        acceleration = quaternion.rotate_vectors(orientation, local_acceleration)

        logging.info('Acceleration from gravity: {}'.format(acceleration))

        acceleration[0] -= 0.005  # I think this is introduced by rounding/precision error
        acceleration[1] -= 0.005
        acceleration[2] -= 0.98  # Get outta here gravity

        deadzone = 0
        if abs(acceleration[0] - .5) <= deadzone:
            acceleration[0] = .5
        if abs(acceleration[1] - .5) <= deadzone:
            acceleration[1] = .5
        if abs(acceleration[2] - .5) <= deadzone:
            acceleration[2] = .5

        # Constrain acceleration to a reasonable level
        max_acceleration_magnitude = 2
        acceleration = np.clip(acceleration, -max_acceleration_magnitude, max_acceleration_magnitude)

        # Convert acceleration to ML-ready 0.0-1.0 values
        acceleration /= max_acceleration_magnitude * 2
        acceleration += 0.5

        self.starting_velocity_estimation_buffer.append((acceleration, microseconds))

        logging.info('Scaled acceleration from movement: {}'.format(acceleration))

        if microseconds == 0:
            frequency = 0
        else:
            frequency = 1 / (microseconds / 1000000)

        if self.bearing_zero is not None:
            # Constrain gesture to a cone
            gesture_cone_angle = 2/3 * np.pi  # 120 degrees

            raw_bearing = bearing

            bearing = np.clip(bearing_delta(self.bearing_zero, bearing),
                              -1/2 * gesture_cone_angle,
                              1/2 * gesture_cone_angle)

            # Scale bearings to ML-ready 0.0-1.0 values
            bearing /= gesture_cone_angle
            bearing += 0.5

        if self.last_bearing_received is not None:
            y, p, r = bearing_delta(self.last_bearing_received, bearing)
            norm = np.sqrt(sum(np.square([abs(y), abs(p)])))

            logging.debug(
                'Last: {} Counter: {} Delta: {}'.format(self.last_bearing_received, bearing, [y, p, r]))
            logging.debug('Norm: {}'.format(norm))

            if norm:
                theta = np.arcsin(norm)
            else:
                theta = 0

            angular_velocity = theta * frequency

            self.angular_velocity_window.append(angular_velocity)

            in_degrees = angular_velocity * 180 / np.pi
            logging.info('Angular velocity {0:.2f} deg or {1:.2f} rad/s'.format(in_degrees, angular_velocity))

        self.last_bearing_received = bearing

        velocity_threshold = .5 if self.state is self.State.recording else 2

        gesture_eligible = (len([x for x in self.angular_velocity_window if x > velocity_threshold])
                            and (fingers == self.pointer_gesture or self.state is self.State.recording))

        motion_scale = 100

        if gesture_eligible:
            if self.state is not self.State.recording:
                self.path_display.delete(ALL)
                self.last_coordinate_visualized = None

                logging.info('Starting sample!')
                self.status_line.configure(bg='SeaGreen1')

                self.bearing_zero = bearing
                self.gesture_hand_position = np.array([0.0, 0.0, 0.0])

                self.gesture_hand_velocity = np.array([0.0, 0.0, 0.0])

                for accel, time in self.starting_velocity_estimation_buffer:
                    self.gesture_hand_velocity += (accel - 0.5) * time / 1000000 * motion_scale

                bearing = np.array([0.5, 0.5, 0.5])

                self.gesture_buffer.append((bearing, local_acceleration, 0))

                self.state = self.State.recording
            else:
                self.gesture_buffer.append((bearing, acceleration, self.gesture_buffer[-1][2] + microseconds))
                self.gesture_hand_velocity += (acceleration - 0.5) * microseconds / 1000000 * motion_scale
                self.gesture_hand_position += self.gesture_hand_velocity * microseconds / 1000000 * motion_scale
                logging.debug('Relative bearing: {0}'.format(str(bearing)))
                logging.debug('Gesture position: ({0:.2f}, {1:.2f}, {2:.2f}'.
                              format(self.gesture_hand_position[0],
                                     self.gesture_hand_position[1],
                                     self.gesture_hand_position[2]))

        else:
            if self.state is self.State.recording:
                # duration = sum([x[2] for x in self.gesture_buffer])
                duration = self.gesture_buffer[-1][2]

                logging.info('Sample done, {} points, {} ms'.format(len(self.gesture_buffer), duration / 1000))
                self.status_line.configure(bg='OliveDrab1')

                # Debug Code!
                # daters = [[q.w, q.x, q.y, q.z] for q in new_gesture.bearings]
                # print(daters)
                #
                # results = self.model.predict([daters], batch_size=1)
                # results = sorted([[chr(ord('a') + i), j] for i, j in enumerate(results[0])],
                #                  key=lambda item: item[1], reverse=True)
                #
                # for index, value in results[:2]:
                #     print('{}: {:.0f}'.format(index, value * 100))
                #
                # if duration > 300000:
                #     self.path_display.create_text((125, 125), text=results[0][0], font=font.Font(family='Comic Sans', size=200))

                def flash_red():
                    self.path_display.configure(bg='salmon1')
                    self.path_display.after(200, lambda: self.path_display.configure(bg='light grey'))

                def flash_blue():
                    self.path_display.configure(bg='SeaGreen1')
                    self.path_display.after(200, lambda: self.path_display.configure(bg='light grey'))

                def overlay_text(text):
                    self.path_display.create_text((5, 245), text=text, anchor=SW)

                if len(self.glyph_picker.selection()) is 0:
                    overlay_text('Discarding - no glyph selected')
                    if duration > 300000:
                        flash_blue()
                    else:
                        flash_red()
                else:
                    selected_glyph = self.glyph_picker.selection()[0]
                    short_glyph = selected_glyph in GestureTrainingSet.short_glyphs

                    if duration > 0 and (short_glyph or duration > 300000):
                        min_yaw = min(sample[0][0] for sample in self.gesture_buffer)
                        max_yaw = max(sample[0][0] for sample in self.gesture_buffer)
                        min_pitch = min(sample[0][1] for sample in self.gesture_buffer)
                        max_pitch = max(sample[0][1] for sample in self.gesture_buffer)

                        if short_glyph or (max_yaw - min_yaw > np.pi / 8 or max_pitch - min_pitch > np.pi / 8):
                            try:
                                processed_bearings, processed_accel = process_samples(self.gesture_buffer,
                                                                                      standard_gesture_time,
                                                                                      sampling_rate)

                                new_gesture = Gesture('', processed_bearings, processed_accel)
                            except AttributeError:
                                logging.info("Couldn't create gesture")
                                new_gesture = None

                            self.training_set.add(new_gesture)

                            self.visualize([x[0] for x in self.gesture_buffer])

                            overlay_text('Accepted sample #{} for glyph {}'.format(
                                self.training_set.count(selected_glyph), selected_glyph))

                            self.path_display.configure(bg='pale green')
                            self.path_display.after(200, lambda: self.path_display.configure(bg='light grey'))

                            self.open_file_has_been_modified = True

                            self.reload_glyph_picker()
                            self.reload_example_list()
                            self.thumbnail_canvas.yview_moveto(1)

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
                            'bearing': bearing,
                            'accel': acceleration,# if self.state is self.State.recording else local_acceleration,
                            'position': self.gesture_hand_position,
                            'freq': frequency})

    def update_status(self, fingers, bearing, accelerometer, frequency):
        def finger(value):
            return '.' if value else '|'

        text = 'Received {0}{1}{2}{3}_, ({4:.2f},{5:.2f},{6:.2f}, ({7:.2f}, {8:.2f}, {9:.2f}) at {10}Hz' \
            .format(finger(fingers[0]), finger(fingers[1]),
                    finger(fingers[2]), finger(fingers[3]),
                    bearing[0], bearing[1], bearing[2],
                    accelerometer[0], accelerometer[1], accelerometer[2],
                    round(frequency))

        self.status_line.configure(text=text, bg='SeaGreen1' if self.state is self.State.recording else 'OliveDrab1')

    def cancel_gesture(self):
        del self.gesture_buffer[:]
        self.bearing_zero = None
        self.gesture_hand_position = None
        self.gesture_hand_velocity = None
        self.last_bearing_received = None
        self.last_coordinate_visualized = None

    def visualize(self, path):
        self.path_display.delete(ALL)

        last_point = None

        for i, coords in enumerate(_path_to_line_points(path, 250, 250, 5, 5)):
            x_coord = coords[0]
            y_coord = coords[1]

            # logging.debug('Plotting ({}, {}) - scaled to ({}, {})'.format(x, y, x_coord, y_coord))

            if last_point:
                green = int(_scale(i, 0, len(path), 255, 0))
                blue = int(_scale(i, 0, len(path), 0, 255))

                self.path_display.create_line(last_point[0],
                                              last_point[1],
                                              x_coord, y_coord,
                                              width=2,
                                              fill='#00{:02X}{:02X}'.format(green, blue))

            last_point = [x_coord, y_coord]

            self.path_display.create_oval((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2), fill='SeaGreen1')


def _path_to_line_points(path, height, width, xpad=0, ypad=0):
    dots = []

    for bearing in path:
        x_coord = np.tan(bearing[1]) * 120 + 122
        y_coord = np.tan(bearing[0]) * -120 + 122
        dots.append([x_coord, y_coord])

    min_x = min([coord[0] for coord in dots])
    max_x = max([coord[0] for coord in dots])
    min_y = min([coord[1] for coord in dots])
    max_y = max([coord[1] for coord in dots])

    output = []

    for x, y in dots:
        output.append((_scale(x, min_x, max_x, xpad, width - xpad * 2),
                       _scale(y, min_y, max_y, ypad, height - ypad * 2)))

    return output


def _scale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def _gesture_to_image(path, height, width, line_thiccness, xpad=0, ypad=0):
    img = Image.new('RGBA', (height, width), (255, 255, 255, 255))
    drawing = ImageDraw.Draw(img)

    scaled_joints = _path_to_line_points(path, height, width, xpad, ypad)

    for i, j in enumerate(scaled_joints):
        if i > 0:
            green = int(_scale(i, 0, len(scaled_joints), 255, 0))
            blue = int(_scale(i, 0, len(scaled_joints), 0, 255))
            drawing.line([scaled_joints[i - 1], j],
                         fill=(0, green, blue, 255),
                         width=line_thiccness)

    # drawing.line(scaled_joints, fill=(0, 0, 255, 255), width=line_thiccness, joint='curve')

    return img
