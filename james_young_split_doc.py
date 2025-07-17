import os
import shutil
from docx import Document

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_PATH = 'JamesYoung.docx'       # Path to your 60 MB .docx
OUTPUT_DIR = 'split_docs'            # Folder where parts will go
MAX_BYTES = 10 * 1024 * 1024         # 10 MB per split
# ──────────────────────────────────────────────────────────────────────────────

def clear_doc_body(doc):
    """
    Remove all existing elements from a Document's body.
    """
    body = doc.element.body
    for child in list(body):
        body.remove(child)

def save_doc(elements, path):
    """
    Create a new .docx containing exactly the given list of XML elements.
    """
    doc = Document()
    clear_doc_body(doc)
    for el in elements:
        # Append the raw XML element to preserve formatting
        doc.element.body.append(el)
    doc.save(path)

def split_docx():
    # 1. Load original and grab all block-level elements
    orig = Document(INPUT_PATH)
    all_elems = list(orig.element.body)  # paragraphs, tables, etc.

    # 2. Prepare output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    current_chunk = []
    part_num = 1

    for el in all_elems:
        current_chunk.append(el)
        # write to a temp file to test size
        temp_path = os.path.join(OUTPUT_DIR, f'temp_{part_num}.docx')
        save_doc(current_chunk, temp_path)
        size = os.path.getsize(temp_path)
        if size > MAX_BYTES:
            # too big—roll back last element
            current_chunk.pop()
            os.remove(temp_path)
            # save the finalized chunk
            final_path = os.path.join(OUTPUT_DIR, f'part_{part_num}.docx')
            save_doc(current_chunk, final_path)
            print(f'→ Saved {final_path} ({os.path.getsize(final_path)//1024} KB)')
            # start a new chunk with the element that overflowed
            part_num += 1
            current_chunk = [el]

    # 3. Save any remaining content
    if current_chunk:
        final_path = os.path.join(OUTPUT_DIR, f'part_{part_num}.docx')
        save_doc(current_chunk, final_path)
        print(f'→ Saved {final_path} ({os.path.getsize(final_path)//1024} KB)')

    # 4. Clean up any temp files
    for fn in os.listdir(OUTPUT_DIR):
        if fn.startswith('temp_'):
            os.remove(os.path.join(OUTPUT_DIR, fn))

if __name__ == '__main__':
    split_docx()
