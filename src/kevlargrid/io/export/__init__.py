"""Data export and executive reporting modules.

Provides high-performance trajectory archives (HDF5), Excel-compatible summary reports (CSV),
off-screen animations (MP4/GIF), and print-ready executive PDF summaries (WeasyPrint).
"""

from __future__ import annotations

from kevlargrid.io.export.csv_writer import export_to_csv
from kevlargrid.io.export.h5_writer import export_to_h5
from kevlargrid.io.export.report_builder import generate_pdf_report, generate_report_html
from kevlargrid.io.export.video_exporter import VideoExporter

__all__ = [
    "VideoExporter",
    "export_to_csv",
    "export_to_h5",
    "generate_pdf_report",
    "generate_report_html",
]
