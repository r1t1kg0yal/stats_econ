#!/usr/bin/env python3
"""
extract_docx_images_gui.py

A script that lets you click “Run” to pick a .docx and an output folder,
then extracts all images and saves them as PNGs.
"""

import zipfile
import os
import io
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

def extract_images(docx_path: str, output_dir: str) -> None:
    """
    Extract images from a .docx file and save them as PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(docx_path, 'r') as docx_zip:
        for file_info in docx_zip.infolist():
            if file_info.filename.startswith('word/media/'):
                data = docx_zip.read(file_info.filename)
                name = os.path.splitext(os.path.basename(file_info.filename))[0]
                try:
                    img = Image.open(io.BytesIO(data))
                except IOError:
                    # not an image
                    continue
                out_path = os.path.join(output_dir, f"{name}.png")
                img.save(out_path, format='PNG')
    messagebox.showinfo("Done", f"Saved images to:\n{output_dir}")

def main():
    root = tk.Tk()
    root.withdraw()  # hide main window

    docx_path = filedialog.askopenfilename(
        title="Select a Word (.docx) file",
        filetypes=[("Word Documents", "*.docx")],
    )
    if not docx_path:
        return  # user cancelled

    output_dir = os.path.join(os.getcwd(), "JamesYoung_images")

    if not output_dir:
        return  # user cancelled

    extract_images(docx_path, output_dir)

if __name__ == "__main__":
    main()
