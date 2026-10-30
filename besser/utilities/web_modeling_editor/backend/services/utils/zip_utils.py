"""Zip archives of generated output, shared by ``/generate-output`` and the traced generation."""

import io
import os
import zipfile


def zip_directory(source_dir: str) -> io.BytesIO:
    """Zip every file under ``source_dir`` into an in-memory archive.

    Entry names are relative to ``source_dir``, so the archive root is the
    directory's content rather than the directory itself.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, source_dir)
                zip_file.write(file_path, arc_name)
    return zip_buffer
