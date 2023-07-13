import tkinter as tk
from tkinter import filedialog, messagebox
from mokuro.run import run
import sys
from loguru import logger

global folder_path
global force_cpu
folder_path = ""
def select_folder():
    global folder_path
    folder_path = filedialog.askdirectory()
    folder_path_label.config(text=folder_path)

def perform_function():
    try:
        if folder_path=="":
            raise ValueError("An error occurred!")
        run(folder_path,force_cpu = force_cpu,disable_confirmation=True)
    except Exception as e:
        messagebox.showerror("Error", str(e))
def toggle_button():
    global force_cpu
    if toggle_var.get() == 0:
        force_cpu = False
        toggle_label.config(text="Using GPU")
    else:
        force_cpu = True
        toggle_label.config(text="Using CPU")

class TkinterHandler:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

root = tk.Tk()
root.title("Folder Select GUI")

text_widget = tk.Text(root)
text_widget.pack()

stdout_redirector = TkinterHandler(text_widget)
logger.add(stdout_redirector.write, format="{message}", catch=False)

# Folder Select Button
folder_btn = tk.Button(root, text="Select Image Folder", command=select_folder)
folder_btn.pack(pady=10)

# Directory Label
folder_path_label = tk.Label(text="")
folder_path_label.pack(pady=10)
folder_path_label.config(text=folder_path)

# GPU Toggle
toggle_var = tk.IntVar()
toggle_button = tk.Checkbutton(root, text="GPU/CPU Toggle", variable=toggle_var, command=toggle_button)
toggle_button.pack(pady=10)
toggle_label = tk.Label(root, text="Using GPU")
toggle_label.pack(pady=5)
force_cpu = True

# Function Button
function_btn = tk.Button(root, text="Perform Function", command=perform_function)
function_btn.pack(pady=10)



root.mainloop()