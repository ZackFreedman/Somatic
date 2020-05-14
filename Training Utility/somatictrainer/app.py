from random import randint
from tkinter import *
from tkinter import messagebox, filedialog, ttk, font

import serial
from serial import Serial, SerialException
from serial.tools.list_ports import comports
import threading
import logging
import requests
import time
from queue import Queue, Empty
from collections import deque
import json
import os
from PIL import Image, ImageTk, ImageDraw
from enum import Enum
import tensorflow.keras as keras
from copy import deepcopy
from somatictrainer.util import *
from somatictrainer.gestures import Gesture, GestureTrainingSet, standard_gesture_length


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

    class TrainingMode(Enum):
        by_glyph = 0
        with_lipsum = 1

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        self.logger = logging.getLogger('HomeWindow')
        self.logger.setLevel(logging.INFO)
        self._log_parsing = True
        self._log_angular_velocity = False

        self.gesture_cone_angle = 2 / 3 * np.pi  # 120 degrees

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
        self.training_mode = self.TrainingMode.with_lipsum

        self.serial_sniffing_thread = None
        self.sample_handling_thread = None

        self.queue = Queue()
        self.port = None
        self.receiving = False
        self.last_hand_id = -1

        self._config_file_version = 1

        self.open_file_pathspec = ''
        self.open_file_has_been_modified = False
        self.change_count_since_last_save = 0
        self.autosave_change_threshold = 25
        self._autosave_timer = None
        self.training_set = GestureTrainingSet()

        self.gesture_buffer = []
        self.raw_data_buffer = []
        self.current_gesture_duration = 0

        self.samples_to_handle = []
        self.bearing_zero = None
        self.last_unprocessed_bearing_received = None
        self.angular_velocity_window = deque(maxlen=10)
        self.starting_velocity_estimation_buffer = deque(maxlen=10)
        self.last_coordinate_visualized = None

        self.lipsum_examples = []
        self.thumbnail_buttons = []

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

        self.training_mode_menu = Menu(self.menu_bar, tearoff=0)

        self.training_mode_menu_selection = IntVar(value=self.training_mode)

        def handle_training_mode_change():
            self.change_training_mode_to(self.TrainingMode(self.training_mode_menu_selection.get()))

        self.training_mode_menu.add_radiobutton(
            label='Train by letter', variable=self.training_mode_menu_selection,
            value=self.TrainingMode.by_glyph.value, command=handle_training_mode_change)
        self.training_mode_menu.add_radiobutton(
            label='Train with sentences', variable=self.training_mode_menu_selection,
            value=self.TrainingMode.with_lipsum.value, command=handle_training_mode_change)
        self.menu_bar.add_cascade(label='Training mode', menu=self.training_mode_menu)

        # This baloney is needed to get the selector menu checkboxes to play nice
        self._serial_port_active_var = BooleanVar()
        self._serial_port_active_var.set(True)
        self._serial_port_inactive_var = BooleanVar()
        self._serial_port_inactive_var.set(False)

        master.config(menu=self.menu_bar)

        left_column = Frame(self, bg='blue')
        right_column = Frame(self, bg='red')

        Label(left_column, text='Hand').pack(fill=X)
        self.hand_display = Canvas(left_column, width=250, height=250)
        self.hand_display.create_image((0, 0), image=self.unknown_hand_icon, anchor=N + W)
        self.hand_display.pack(fill=X)

        Label(left_column, text='Path').pack(fill=X)
        self.path_display = Canvas(left_column, width=250, height=250, bg='white')
        self.path_display.pack(fill=X)

        label_width_locking_frame = Frame(right_column, height=20, width=250)
        label_width_locking_frame.grid(row=0, column=0, sticky=N + E + W)
        self.file_name_label = Label(label_width_locking_frame, text='No training file open', bg='white', relief=SUNKEN)
        self.file_name_label.pack(fill=BOTH, expand=1)

        picker_frame = Frame(right_column, bd=2, relief=SUNKEN)
        picker_frame.grid_columnconfigure(0, minsize=284)

        picker_frame.grid(row=2, column=0, sticky=S + E + W)

        self.thumbnail_canvas = Canvas(picker_frame, bd=0, width=250, height=150, bg='white')
        self.thumbnail_canvas.grid(row=0, column=0, sticky=N + S + E + W)

        self.thumbnail_scrollbar = Scrollbar(picker_frame)
        self.thumbnail_scrollbar.grid(row=0, column=1, sticky=N + S)

        self.thumbnail_canvas.configure(yscrollcommand=self.thumbnail_scrollbar.set)
        self.thumbnail_scrollbar.config(command=self.thumbnail_canvas.yview)
        self.thumbnail_canvas.config(yscrollcommand=self.thumbnail_scrollbar.set)

        def bind_wheel_to_thumbnails(event):
            self.thumbnail_canvas.bind_all('<MouseWheel>', on_wheel_scroll)

        def unbind_wheel_from_thumbnails(event):
            self.thumbnail_canvas.unbind_all('<MouseWheel>')

        def on_wheel_scroll(event):
            self.thumbnail_canvas.yview_scroll(int(event.delta / -120), 'units')

        self.thumbnail_frame = Frame(self.thumbnail_canvas, bg='white')
        self.thumbnail_frame.bind('<Enter>', bind_wheel_to_thumbnails)
        self.thumbnail_frame.bind('<Leave>', unbind_wheel_from_thumbnails)
        self.thumbnail_frame.bind('<Configure>', lambda x: self.thumbnail_canvas.configure(
            scrollregion=self.thumbnail_canvas.bbox(ALL)))

        self.thumbnail_canvas.create_window((0, 0), window=self.thumbnail_frame, anchor=NW)

        for i in range(5):
            self.thumbnail_frame.grid_columnconfigure(i, weight=1)

        self.glyph_picker = ttk.Treeview(right_column, column='count')
        self.glyph_picker.column("#0", width=100, stretch=False)
        self.glyph_picker.column('count', stretch=True)
        self.glyph_picker.heading('count', text='Count')
        self.glyph_picker.grid(row=1, column=0, sticky=N + S + E + W)
        if self.training_mode is not self.TrainingMode.by_glyph:
            self.glyph_picker.grid_remove()

        self.glyph_picker.bind('<<TreeviewSelect>>', lambda x: self.reload_example_list())

        self.lipsum_text = Text(right_column, font=font.Font(family='Comic Sans MS', size=18), wrap=WORD)
        self.lipsum_text.tag_config('neutral', background='white', foreground='black')
        self.lipsum_text.tag_config('correct', background='palegreen', foreground='darkgreen')
        self.lipsum_text.tag_config('incorrect', background='lightcoral', foreground='darkred')
        self.lipsum_text.tag_config('selected', background='black', foreground='white')

        self.reset_lipsum()

        self.lipsum_text.grid(row=1, column=0, sticky=N + S + E + W)
        if self.training_mode is not self.TrainingMode.with_lipsum:
            self.lipsum_text.grid_remove()

        self.reload_glyph_picker()  # This must be here - self.lipsum_text must exist if we start in lipsum mode

        self.status_line = Label(self, text='Not connected', bd=1, relief=SUNKEN, anchor=W, bg='light goldenrod')

        left_column.grid(row=0, column=0, sticky=N)
        right_column.grid(row=0, column=1, sticky=N + S + E + W)
        right_column.grid_propagate(False)
        right_column.grid_columnconfigure(0, weight=1, minsize=250)
        right_column.grid_rowconfigure(1, weight=1)
        self.status_line.grid(row=1, column=0, columnspan=2, sticky=S + E + W)

        self.grid_columnconfigure(0, weight=0, minsize=250)
        self.grid_columnconfigure(1, weight=1, minsize=250)
        self.grid_rowconfigure(0, weight=1, minsize=500)

        # self.grid_propagate(True)

        # Debug code!
        self.model = keras.models.load_model(
            'E:\\Dropbox\\Projects\\Source-Controlled Projects\\Somatic\Training Utility\\training_set_2_bak.h5')

        # self.model = None

    def save_state(self):
        datastore = {'port': self.port.port if self.port is not None and self.port.isOpen() else None,
                     'wip': self.open_file_pathspec,
                     'selected-glyph': self.get_selected_glyph(),
                     'version': self._config_file_version}

        with open(os.getcwd() + 'config.json', 'w') as f:
            json.dump(datastore, f)

    def restore_state(self):
        if not os.path.isfile(os.getcwd() + 'config.json'):
            self.logger.warning('No saved configuration, starting fresh')
            return

        with open(os.getcwd() + 'config.json', 'r') as f:
            datastore = json.load(f)

            if 'version' not in datastore or datastore['version'] != self._config_file_version:
                self.logger.warning('Saved config is outdated, not loading')
                return

            if 'wip' in datastore:
                self.logger.info('Re-opening work-in-progress file {}'.format(datastore['wip']))

                if self.open_file(datastore['wip']):
                    try:
                        if 'selected-glyph' in datastore and datastore['selected-glyph']:
                            selected_item = datastore['selected-glyph']
                            if selected_item is not None:
                                self.glyph_picker.focus(selected_item)
                                self.glyph_picker.selection_set(selected_item)
                                self.glyph_picker.see(selected_item)
                            self.logger.info('Resuming with glyph {} selected'.format(datastore['selected-glyph']))
                    except TclError:
                        self.logger.warning("Couldn't select glyph {} - it doesn't exist?".format(datastore['selected-glyph']))
                else:
                    self.logger.warning("Couldn't reopen file - doesn't exist?")

            if 'port' in datastore and datastore['port']:
                port_names = [port.device for port in comports()]

                if datastore['port'] in port_names:
                    self.disconnect()
                    self.connect_to(datastore['port'])
                    self.logger.info('Reconnected to glove on {}'.format(datastore['port']))
                else:
                    self.logger.info("Couldn't reconnect to port {}".format(datastore['port']))

    def reset_lipsum(self):
        # Get rid of examples from the previous sentence
        del self.lipsum_examples[:]
        if self.training_mode is self.TrainingMode.with_lipsum:
            self.reload_example_list()

        # Get Hipster Ipsum, strip off trailing punctuation
        call = requests.get('https://hipsum.co/api/?type=hipster-centric&sentences=1').json()[0]
        tokens = call.split(' ')

        for i in range(len(tokens)):
            if randint(0, 2) == 0:
                # Randomly capitalize some of the words to collect more caps samples
                tokens[i] = tokens[i].capitalize()
            elif randint(0, 5) == 0:
                tokens[i] = tokens[i].upper()
            if i < len(tokens) - 1:
                tokens[i] += ' '

        number = ''

        for i in range(len(tokens)):
            if randint(0, 4) == 0:
                # Also add some numbers, we need more of those too
                for j in range(2, 6):
                    number += chr(ord('0') + randint(0, 9))
                tokens.insert(i, number + ' ')
                number = ''

        symbols = '!"#$\',-./?@'  # No spaces - we got so many fukken spaces. The best spaces. The emptiest pixels.

        for i in range(len(tokens)):
            if randint(0, 4) == 0:
                if i > 0:
                    tokens[i - 1] = tokens[i - 1][:-1]  # Get rid of that space to make it clearer and easier to type
                # Add some punctuation symbols too
                tokens.insert(i, symbols[randint(0, len(symbols) - 1)] + ' ')

        hipsum = ''.join(tokens)  # Put it all together

        self.lipsum_text.config(state=NORMAL)  # This is required to edit the text
        self.lipsum_text.delete('1.0', END)
        self.lipsum_text.insert(END, hipsum)
        self.lipsum_text.tag_add('selected', '1.0')
        self.lipsum_text.config(state=DISABLED)  # This is required to prevent the user from editing the lipsum

    def reload_glyph_picker(self):
        scroll_position = self.glyph_picker.yview()[0]
        selected_item = self.get_selected_glyph()

        self.glyph_picker.delete(*self.glyph_picker.get_children())

        for glyph in GestureTrainingSet.big_ole_list_o_glyphs:
            if glyph is '\x08':
                glyph_string = '<Backspace>'
            elif glyph is '\n':
                glyph_string = '<Return>'
            elif glyph is ' ':
                glyph_string = '<Space>'
            else:
                glyph_string = glyph

            self.glyph_picker.insert('', 'end', text="Glyph '{}'".format(glyph_string),
                                     value=str(self.training_set.count(glyph)) if self.training_set else '0',
                                     iid=glyph)

        if selected_item is not None:
            self.glyph_picker.focus(selected_item)
            self.glyph_picker.selection_set(selected_item)
            self.glyph_picker.yview_moveto(scroll_position)

    def get_selected_glyph(self):
        if self.training_mode is self.TrainingMode.by_glyph:
            return self.glyph_picker.selection()[0] if len(self.glyph_picker.selection()) else None
        elif self.training_mode is self.TrainingMode.with_lipsum:
            selection_range = self.lipsum_text.tag_nextrange('selected', '1.0')
            return self.lipsum_text.get(selection_range[0], selection_range[1])
        else:
            return None

    def reload_example_list(self):
        for button in self.thumbnail_buttons:
            button.grid_forget()
            button.destroy()

        del self.thumbnail_buttons[:]

        if self.training_mode is self.TrainingMode.by_glyph:
            selected_glyph = self.get_selected_glyph()
            if selected_glyph is None or not self.training_set.count(selected_glyph):
                return

            for example in self.training_set.get_examples_for(selected_glyph):
                self.insert_thumbnail_button_for(example)

        elif self.training_mode is self.TrainingMode.with_lipsum:
            for example in self.lipsum_examples:
                self.insert_thumbnail_button_for(example)

        self.thumbnail_canvas.yview_moveto(0)

    def insert_thumbnail_button_for(self, example):
        for button in self.thumbnail_buttons:
            if button.gesture == example:
                self.logger.debug('Already placed button for example w/ UUID {}'.format(example.uuid))
                return

        if example in self.example_thumbnails:
            thumbnail = self.example_thumbnails[example]
        else:
            thumbnail = ImageTk.PhotoImage(
                image=_gesture_to_image(
                    example.bearings, 50, 50, 2, 2, 2))
            self.example_thumbnails[example] = thumbnail

        button = Button(self.thumbnail_frame, image=thumbnail)
        button.image = thumbnail  # Button doesn't have an image field - this monkey patch retains a reference
        button.gesture = example  # Monkey-patch a reference to the gesture into the button to associate them
        button.configure(command=lambda: self.visualize(example.bearings))

        # Right click on OSX
        button.bind('<Button-2>', lambda x: self.delete_thumbnail_button(button))
        # Right click on Windows
        button.bind('<Button-3>', lambda x: self.delete_thumbnail_button(button))

        self.thumbnail_buttons.append(button)
        position = len(self.thumbnail_buttons) - 1
        button.grid(row=int(position / 5), column=int(position % 5))

        if len(self.thumbnail_buttons) == 1:
            # This is the first thumbnail in the list - the scrollbar won't readjust automatically.
            self.thumbnail_canvas.yview_moveto(0)

    def delete_thumbnail_button(self, button):
        self.logger.info('Removing example for {}, UUID {}'.format(
            button.gesture.glyph, str(button.gesture.uuid)))
        self.training_set.remove(button.gesture)

        # self.reload_example_list()
        position = self.thumbnail_buttons.index(button)
        button.grid_forget()
        button.destroy()
        self.thumbnail_buttons.remove(button)

        for i in range(position, len(self.thumbnail_buttons)):
            self.thumbnail_buttons[i].grid(row=int(i / 5), column=int(i % 5))

        self.glyph_picker.item(button.gesture.glyph, value=self.training_set.count(button.gesture.glyph))

        # We can save now! Yay!
        self.open_file_has_been_modified = True
        self.file_menu.entryconfigure(self.save_entry_index, state=NORMAL)
        self.file_menu.entryconfigure(self.save_as_entry_index, state=NORMAL)

        if self.open_file_pathspec:
            self.change_count_since_last_save += 1  # Autosave. Make Murphy proud.
            if self.change_count_since_last_save >= self.autosave_change_threshold:
                self.plan_autosave()

    def plan_autosave(self):
        if self._autosave_timer is not None:
            self.after_cancel(self._autosave_timer)

        def autosave():
            self.save_file()
            self.change_count_since_last_save = 0
            self._autosave_timer = None

        # self._autosave_timer = self.after(30000, autosave)
        self._autosave_timer = self.after(10, autosave)

    def start(self):
        self.master.after(10, self.queue_handler)
        self.restore_state()
        self.logger.debug('Gesture cone angle: {:.5f}'.format(self.gesture_cone_angle))# TODO cut this

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

        self.save_state()

        self.queue.put({'type': 'quit'})

    def queue_handler(self):
        try:
            command = self.queue.get(block=False)

            if command['type'] is 'ack':
                if self.state is self.State.connecting:
                    self.state = self.State.connected
                    self.logger.info('Got ack - connected')

            if command['type'] is 'viz':
                path = command['path']

                self.path_display.delete(ALL)

                last_point = None

                x_center = (max(path[:, 0] - min(path[:, 0]))) / 2
                y_center = (max(path[:, 1] - min(path[:, 1]))) / 2

                padding = 10

                for i, coords in enumerate(path):
                    x_coord = (coords[0] + 0.5 - x_center) * (250 - padding * 2) + padding
                    y_coord = (coords[1] + 0.5 - y_center) * (250 - padding * 2) + padding

                    if last_point:
                        green = int(_scale(i, 0, len(path), 255, 0))
                        blue = int(_scale(i, 0, len(path), 0, 255))

                        self.path_display.create_line(last_point[0],
                                                      last_point[1],
                                                      x_coord, y_coord,
                                                      width=2,
                                                      fill='#00{:02X}{:02X}'.format(green, blue))

                    last_point = [x_coord, y_coord]

                    self.path_display.create_oval((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2),
                                                  fill='SeaGreen1')

            if command['type'] is 'rx':
                bearing = command['bearing']
                if self.state is not self.State.disconnected and self.state is not self.State.quitting:
                    self.update_status(command['fingers'], bearing, command['freq'])

                    fingers = command['fingers']
                    hand_id = fingers[0] * 0b1000 + fingers[1] * 0b0100 + fingers[2] * 0b0010 + fingers[3] * 0b0001
                    if hand_id != self.last_hand_id:
                        self.hand_display.create_image((0, 0), image=self.hand_icons[hand_id], anchor=N + W)
                        self.last_hand_id = hand_id

                if self.state is self.State.recording:
                    x_coord = np.tan(bearing[0]) * 250
                    y_coord = np.tan(bearing[1]) * 250

                    if 0 <= x_coord <= 250 and 0 <= y_coord <= 250:
                        # self.logger.debug('Rendering ({}, {})'.format(x_coord, y_coord))

                        if self.last_coordinate_visualized:
                            self.path_display.create_line(self.last_coordinate_visualized[0],
                                                          self.last_coordinate_visualized[1],
                                                          x_coord, y_coord,
                                                          width=2, fill='blue')

                        self.last_coordinate_visualized = [x_coord, y_coord]

                        self.path_display.create_oval((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2),
                                                      fill='SeaGreen1')
                    else:
                        self.logger.debug('Coordinate ({}, {}) invalid - leftover from before gesture?'
                                          .format(x_coord, y_coord))

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
            filetypes=(('Database file', '*.db'), ('All files', '*')))
        if new_file_pathspec:
            new_file_pathspec += '.db'
            self.open_file_pathspec = new_file_pathspec
            open(new_file_pathspec, 'w')  # Save empty file

            self.training_set = GestureTrainingSet()

            self.file_menu.entryconfigure(self.save_entry_index, state=NORMAL)
            self.file_menu.entryconfigure(self.save_as_entry_index, state=NORMAL)
            self.file_name_label.configure(text=self.open_file_pathspec, anchor=E)

            self.open_file_has_been_modified = True
            self.reload_glyph_picker()

    def open_file(self, file_to_open=None):
        if file_to_open is None:
            file_to_open = filedialog.askopenfilename(title='Select training data',
                                                      filetypes=(('Database file', '*.db'), ('All files', '*')))

        if not os.path.isfile(file_to_open):
            self.logger.warning("Can't open training file because it doesn't exist - {}".format(file_to_open))
            return

        if file_to_open:
            try:
                training_set = GestureTrainingSet.load(file_to_open)
                self.training_set = training_set
                self.reload_glyph_picker()

                self.open_file_pathspec = file_to_open
                self.file_menu.entryconfigure('Save', state=NORMAL)
                self.file_menu.entryconfigure('Save as...', state=NORMAL)
                self.file_name_label.configure(text=self.open_file_pathspec, anchor=E)
                return True
            except (KeyError, AttributeError, json.decoder.JSONDecodeError) as e:
                self.logger.exception('Oopsie')
                messagebox.showerror("Can't load file",
                                     "This file can't be loaded.\nError: {}".format(repr(e)))
                return False

    def save_file(self):
        if self.state is self.State.recording:
            self.state = self.State.connected
            self.cancel_gesture()

        if not self.training_set:
            self.logger.warning('Tried to save empty training set?')
            return

        if not self.open_file_pathspec:
            file_to_create = filedialog.asksaveasfilename(title='Save training data',
                                                          filetypes=(('JSON dictionary', '*.json'), ('All files', '*')))
            if file_to_create:
                self.open_file_pathspec = file_to_create + '.db'
            else:
                return

        self.training_set.save(self.open_file_pathspec)

        self.logger.info('Saved {}'.format(self.open_file_pathspec))

        self.open_file_has_been_modified = False
        self.change_count_since_last_save = 0

    def save_as(self):
        if self.state is self.State.recording:
            self.state = self.State.connected
            self.cancel_gesture()

        if not self.training_set:
            self.logger.warning('Tried to save empty training set?')
            return

        file_to_create = filedialog.asksaveasfilename(title='Save training data',
                                                      filetypes=(('Database file', '*.db'), ('All files', '*')))
        if file_to_create:
            self.open_file_pathspec = file_to_create
        else:
            return

        self.training_set.save(self.open_file_pathspec)

        self.open_file_has_been_modified = False

    def populate_serial_port_menu(self):
        self.port_menu.delete(0, 999)

        ports = comports()
        self._serial_port_active_var.set(True)
        self._serial_port_inactive_var.set(False)

        if ports:
            for path, _, __ in ports:
                if self.port and self.port.isOpen() and path == self.port.port:
                    self.port_menu.add_checkbutton(label=path, command=self.disconnect,
                                                   variable=self._serial_port_active_var, onvalue=True, offvalue=False)
                else:
                    self.port_menu.add_checkbutton(label=path, command=lambda pathspec=path: self.connect_to(pathspec),
                                                   variable=self._serial_port_inactive_var, onvalue=True, offvalue=False)

        else:
            self.port_menu.add_command(label='No serial ports available', state=DISABLED)

    def change_training_mode_to(self, new_mode):
        self.training_mode_menu_selection.set(new_mode)

        if new_mode is self.TrainingMode.by_glyph:
            if self.training_mode is not self.TrainingMode.by_glyph:
                logging.info('Changing to train-by-glyph mode')
                self.training_mode = self.TrainingMode.by_glyph
                self.lipsum_text.grid_remove()
                self.glyph_picker.grid()
                self.reload_example_list()
            else:
                logging.info('Already training by glyph')

        elif new_mode is self.TrainingMode.with_lipsum:
            if self.training_mode is not self.TrainingMode.with_lipsum:
                logging.info('Changing to train-by-sentence mode')
                self.training_mode = self.TrainingMode.with_lipsum
                self.glyph_picker.grid_remove()
                self.lipsum_text.grid()
                self.reload_example_list()
            else:
                logging.info('Already training with lipsum')

        else:
            raise AttributeError('Unrecognized training mode {}'.format(new_mode))

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

        except SerialException as e:
            self.logger.exception('dafuq')
            error_tokens = [x.strip("' ") for x in e.args[0].split('(')[1].split(',')]
            messagebox.showinfo("Can't open {}".format(portspec), 'Error {}: {}'.format(error_tokens[0], error_tokens[1]))
            self.state = self.State.disconnected
            self.status_line.configure(text="Can't connect to {}".format(portspec), bg='firebrick1')
            self.port = None

    def disconnect(self):
        if self.port:
            self.logger.info('Now disconnecting')

            if self.receiving:
                self.receiving = False
                if threading.current_thread() is not self.serial_sniffing_thread \
                        and self.serial_sniffing_thread \
                        and self.serial_sniffing_thread.is_alive():
                    self.logger.info('Quitting receive')
                    self.serial_sniffing_thread.join(timeout=1)
                    self.logger.info('No longer receiving')

            if self.port.isOpen():
                self.logger.info('Port closed')
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
            self.logger.warning('Already receiving, not doing it')
            return

        self.receiving = True
        self.logger.info('Now receiving')

        self.serial_sniffing_thread = threading.Thread(target=self.handle_packets, daemon=True)
        self.serial_sniffing_thread.start()

    def handle_packets(self):
        while self.receiving and self.port and self.port.isOpen():
            try:
                incoming = self.port.readline().decode()
            except SerialException:
                self.logger.exception('Lost serial connection, bailing out')
                self.disconnect()
                return

            if incoming:
                if self._log_parsing:
                    self.logger.debug('Received packet {}'.format(incoming))

                # Packet format:
                # >[./|],[./|],[./|],[./|],
                # [float o.h],[float o.p],[float o.r],
                # [float a.x], [float a.y], [float a.z], [us since last sample]

                if incoming.count('>') is not 1:
                    if self._log_parsing:
                        self.logger.debug('Packet corrupt - missing delimeter(s)')
                    continue

                # Strip crap that arrived before the delimeter, and also the delimiter itself
                incoming = incoming[incoming.index('>') + 1:].rstrip()

                if incoming == 'OK':
                    self.logger.info('Received ack')
                    self.queue.put({'type': 'ack'})
                    continue

                if incoming.count(',') is not 10:
                    if self._log_parsing:
                        self.logger.debug('Packet corrupt - insufficient fields')
                    continue

                tokens = incoming.split(',')
                if self._log_parsing:
                    self.logger.debug('Tokens: {0}'.format(tokens))

                fingers = list(map(lambda i: i is '.', tokens[:4]))
                if self._log_parsing:
                    self.logger.debug('Fingers: {0}'.format(fingers))

                bearing = np.array([float(t) for t in tokens[4:7]])
                acceleration = np.array([float(t) for t in tokens[7:10]])

                microseconds = float(tokens[-1])

                if self._log_parsing:
                    self.logger.debug('Sample parsed')

                self.accept_sample(fingers, bearing, acceleration, microseconds)

            else:
                time.sleep(0.05)

    def accept_sample(self, *args):
        self.samples_to_handle.append(args)

        if not self.sample_handling_thread or not self.sample_handling_thread.is_alive():
            self.sample_handling_thread = threading.Thread(target=self.sample_handling_loop, daemon=True)
            self.sample_handling_thread.start()

    def sample_handling_loop(self):
        while len(self.samples_to_handle):
            fingers, bearing, acceleration, microseconds = self.samples_to_handle[0]
            del self.samples_to_handle[0]
            self.handle_sample(fingers, bearing, acceleration, microseconds)
            time.sleep(0)

    def handle_sample(self, fingers, bearing, acceleration, microseconds):
        """

        :type fingers: list of bool
        :type bearing: np.array
        :type acceleration: np.array
        :type microseconds: float
        """

        if microseconds == 0:
            frequency = 0
        else:
            frequency = 1 / (microseconds / 1000000)

        raw_bearing = bearing

        bearing = np.array([bearing[0], bearing[1]])  # We don't care about roll

        if self.last_unprocessed_bearing_received is not None:
            y, p = bearing_delta(self.last_unprocessed_bearing_received, bearing)
            norm = np.sqrt(sum(np.square([abs(y), abs(p)])))

            if self._log_angular_velocity:
                self.logger.debug(
                    'Last: {} Counter: {} Delta: {}'.format(self.last_unprocessed_bearing_received, bearing, [y, p]))
                self.logger.debug('Norm: {}'.format(norm))

            if norm:
                theta = np.arcsin(norm)
            else:
                theta = 0

            angular_velocity = theta * frequency

            self.angular_velocity_window.append(angular_velocity)

            in_degrees = angular_velocity * 180 / np.pi

            if self._log_angular_velocity:
                self.logger.debug('Angular velocity {0:.2f} deg or {1:.2f} rad/s'.format(in_degrees, angular_velocity))

        self.last_unprocessed_bearing_received = bearing

        velocity_threshold = 2 if self.state is self.State.recording else 4

        gesture_eligible = (len([x for x in self.angular_velocity_window if x > velocity_threshold])
                            and (fingers == self.pointer_gesture or self.state is self.State.recording))

        if self.bearing_zero is not None:
            # Constrain gesture to a cone
            self.logger.debug('Raw bearing: ({:.3f}, {:.3f})'.format(bearing[0], bearing[1]))

            bearing = bearing_delta(self.bearing_zero, bearing)
            # self.logger.debug('Processed bearing 1: ({:.3f}, {:.3f})'.format(bearing[0], bearing[1]))

            bearing = np.clip(bearing,
                              -1 / 2 * self.gesture_cone_angle,
                              1 / 2 * self.gesture_cone_angle)
            # self.logger.debug('Processed bearing 2: ({:.3f}, {:.3f})'.format(bearing[0], bearing[1]))

            # Scale bearings to ML-ready 0.0-1.0 values
            bearing /= self.gesture_cone_angle
            # self.logger.debug('Processed bearing 3: ({:.3f}, {:.3f})'.format(bearing[0], bearing[1]))

            bearing += 0.5
            # self.logger.debug('Processed bearing 4: ({:.3f}, {:.3f})'.format(bearing[0], bearing[1]))

            self.logger.debug('Processed bearing: ({:.3f}, {:.3f})'.format(bearing[0], bearing[1]))

        if gesture_eligible:
            if self.state is not self.State.recording:
                self.path_display.delete(ALL)
                self.last_coordinate_visualized = None

                self.logger.info('Starting sample!')
                self.status_line.configure(bg='SeaGreen1')

                self.bearing_zero = bearing
                self.logger.debug('Bearing zero is ({:.3f}, {:.3f})'.format(self.bearing_zero[0], self.bearing_zero[1]))

                bearing = np.array([0.5, 0.5])

                self.gesture_buffer.append(bearing)

                self.state = self.State.recording
            else:
                self.gesture_buffer.append(bearing)

            self.raw_data_buffer.append({'b': raw_bearing.tolist(), 'a': acceleration.tolist(), 't': microseconds})
            self.current_gesture_duration += microseconds

        else:
            if self.state is self.State.recording:
                self.logger.info('Sample done, {} points'.format(len(self.gesture_buffer)))
                self.status_line.configure(bg='OliveDrab1')

                def flash_red():
                    self.path_display.configure(bg='salmon1')
                    self.path_display.after(200, lambda: self.path_display.configure(bg='light grey'))

                def flash_blue():
                    self.path_display.configure(bg='SeaGreen1')
                    self.path_display.after(200, lambda: self.path_display.configure(bg='light grey'))

                def overlay_text(text):
                    self.path_display.create_text((5, 245), text=text, anchor=SW)

                try:
                    processed_bearings = process_samples(np.array(self.gesture_buffer), standard_gesture_length)

                    for index, bearing in enumerate(processed_bearings):
                        self.logger.debug("Point {}: ({:.4f}, {:.2f})".format(index, bearing[0], bearing[1]))

                    new_gesture = Gesture('', processed_bearings, deepcopy(self.raw_data_buffer))

                    self.visualize(processed_bearings)
                except AttributeError:
                    self.logger.exception("Couldn't create gesture")
                    new_gesture = None

                if new_gesture and len(self.glyph_picker.selection()) is 0:
                    overlay_text('Discarding - no glyph selected')

                    if self.current_gesture_duration > 300000:
                        flash_blue()
                    else:
                        flash_red()
                elif new_gesture:
                    daters = new_gesture.bearings

                    # results = self.model.predict(daters.reshape(1, standard_gesture_length, 2))
                    results = self.model.predict(daters.reshape(1, standard_gesture_length * 2))
                    # self.logger.info(len(results[0]))
                    winning_glyph, confidence = \
                    sorted([[self.training_set.get_character_map()[i], j] for i, j in enumerate(results[0])],
                           key=lambda item: item[1], reverse=True)[0]

                    selected_glyph = self.get_selected_glyph()

                    logging.info('Predicted \'{}\' with {:.2f}% confidence - {}!'.format(
                        '0x' + hex(winning_glyph) if ord(winning_glyph) < ord(' ') else winning_glyph,
                        confidence * 100, 'CORRECT' if winning_glyph == selected_glyph else 'WRONG'))

                    if confidence > 0.95:
                        self.path_display.create_text((125, 125),
                                                      text='0x' + hex(winning_glyph)
                                                      if ord(winning_glyph) < ord(' ')
                                                      else winning_glyph,
                                                      font=font.Font(family='Comic Sans MS', size=200))
                    else:
                        self.path_display.create_text((125, 125), text='?',
                                                      font=font.Font(family='Comic Sans MS', size=200))
                        logging.info('Too little confidence in this result to definitively call it')
                        winning_glyph = None

                    short_glyph = selected_glyph in GestureTrainingSet.short_glyphs

                    if self.current_gesture_duration > 0 and (short_glyph or self.current_gesture_duration > 300000):
                        if winning_glyph == selected_glyph:
                            flash_blue()
                        else:
                            flash_red()

                        # min_yaw = min(sample[0] for sample in self.gesture_buffer)
                        # max_yaw = max(sample[0] for sample in self.gesture_buffer)
                        # min_pitch = min(sample[1] for sample in self.gesture_buffer)
                        # max_pitch = max(sample[1] for sample in self.gesture_buffer)
                        #
                        # tiniest_glyph = 1 / 32 * np.pi

                        # if short_glyph or (max_yaw - min_yaw > tiniest_glyph or max_pitch - min_pitch > tiniest_glyph):
                        new_gesture.glyph = selected_glyph
                        self.training_set.add(new_gesture)

                        overlay_text('Accepted sample #{} for glyph {}'.format(
                            self.training_set.count(selected_glyph), selected_glyph))

                        # TODO make this go red if it recognized wrong
                        self.path_display.configure(bg='pale green')
                        self.path_display.after(200, lambda: self.path_display.configure(bg='light grey'))

                        # We can save now! Yay!
                        self.open_file_has_been_modified = True
                        self.file_menu.entryconfigure(self.save_entry_index, state=NORMAL)
                        self.file_menu.entryconfigure(self.save_as_entry_index, state=NORMAL)

                        if self.open_file_pathspec:
                            self.change_count_since_last_save += 1  # Autosave. Make Murphy proud.
                            if self.change_count_since_last_save >= self.autosave_change_threshold:
                                self.plan_autosave()

                        self.insert_thumbnail_button_for(new_gesture)
                        self.glyph_picker.item(new_gesture.glyph,
                                               value=self.training_set.count(new_gesture.glyph))
                        self.thumbnail_canvas.yview_moveto(1)

                        if self.training_mode is self.TrainingMode.with_lipsum:
                            selection_index = self.lipsum_text.tag_nextrange('selected', '1.0')[0]
                            self.lipsum_text.tag_remove('selected', selection_index)

                            if winning_glyph == selected_glyph:
                                self.lipsum_text.tag_add('correct', selection_index)
                            else:
                                self.lipsum_text.tag_add('incorrect', selection_index)

                            selection_index_position = int(selection_index.split('.')[-1])

                            if selection_index_position >= len(self.lipsum_text.get('1.0', END)):
                                # We've completed the sentence! Leave it onscreen a bit for maximum satisfaction.
                                self.logger.info('Finished sentence! Well done.')
                                self.after(2000, self.reset_lipsum)
                            else:
                                # Advance the thingy!
                                self.lipsum_text.tag_add('selected', '1.{}'.format(selection_index_position + 1))

                        # else:
                        #     overlay_text('Discarding - too small')
                        #     flash_red()
                    else:
                        overlay_text('Discarding - too short')
                        flash_red()

                self.cancel_gesture()
                self.state = self.State.connected

        if self.queue.empty():
            self.queue.put({'type': 'rx',
                            'fingers': fingers,
                            'bearing': bearing,
                            'freq': frequency})

    def update_status(self, fingers, bearing, frequency):
        def finger(value):
            return '.' if value else '|'

        text = 'Received {}{}{}{}_, ({:.2f},{:.2f}) at {}Hz' \
            .format(finger(fingers[0]), finger(fingers[1]),
                    finger(fingers[2]), finger(fingers[3]),
                    bearing[0], bearing[1],
                    round(frequency))

        self.status_line.configure(text=text, bg='SeaGreen1' if self.state is self.State.recording else 'OliveDrab1')

    def cancel_gesture(self):
        del self.gesture_buffer[:]
        del self.raw_data_buffer[:]
        self.current_gesture_duration = 0
        self.bearing_zero = None
        self.last_unprocessed_bearing_received = None
        self.last_coordinate_visualized = None

    def visualize(self, path):
        self.queue.put({'type': 'viz', 'path': path})


def _scale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def _gesture_to_image(path, height, width, line_thiccness, xpad=0, ypad=0):
    img = Image.new('RGBA', (height, width), (255, 255, 255, 255))
    drawing = ImageDraw.Draw(img)

    x_center = (max(path[:, 0] - min(path[:, 0]))) / 2
    y_center = (max(path[:, 1] - min(path[:, 1]))) / 2

    for i, coords in enumerate(path):
        if i > 0:
            prev_x_coord = (path[i - 1][0] + 0.5 - x_center) * (width - xpad * 2)
            prev_y_coord = (path[i - 1][1] + 0.5 - y_center) * (height - ypad * 2)
            x_coord = (coords[0] + 0.5 - x_center) * (width - xpad * 2)
            y_coord = (coords[1] + 0.5 - y_center) * (height - ypad * 2)

            green = int(_scale(i, 0, len(path), 255, 0))
            blue = int(_scale(i, 0, len(path), 0, 255))
            drawing.line(((prev_x_coord, prev_y_coord), (x_coord, y_coord)),
                         fill=(0, green, blue, 255),
                         width=line_thiccness)

    # drawing.line(scaled_joints, fill=(0, 0, 255, 255), width=line_thiccness, joint='curve')

    return img
