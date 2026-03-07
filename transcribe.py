import sys
import os
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QListWidget, QLabel,
    QFileDialog, QMessageBox, QStatusBar, QFrame,
    QComboBox, QLineEdit, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QIntValidator, QTextCursor

import whisper_engine  # Import our engine file


# ── Whisper supported languages (display name → Whisper language code) ────────

WHISPER_LANGUAGES: dict[str, str | None] = {
    "Auto Detect":    None,
    "Afrikaans":      "af",
    "Albanian":       "sq",
    "Amharic":        "am",
    "Arabic":         "ar",
    "Armenian":       "hy",
    "Assamese":       "as",
    "Azerbaijani":    "az",
    "Bashkir":        "ba",
    "Basque":         "eu",
    "Belarusian":     "be",
    "Bengali":        "bn",
    "Bosnian":        "bs",
    "Breton":         "br",
    "Bulgarian":      "bg",
    "Burmese":        "my",
    "Catalan":        "ca",
    "Chinese":        "zh",
    "Croatian":       "hr",
    "Czech":          "cs",
    "Danish":         "da",
    "Dutch":          "nl",
    "English":        "en",
    "Estonian":       "et",
    "Faroese":        "fo",
    "Finnish":        "fi",
    "French":         "fr",
    "Galician":       "gl",
    "Georgian":       "ka",
    "German":         "de",
    "Greek":          "el",
    "Gujarati":       "gu",
    "Haitian Creole": "ht",
    "Hausa":          "ha",
    "Hawaiian":       "haw",
    "Hebrew":         "he",
    "Hindi":          "hi",
    "Hungarian":      "hu",
    "Icelandic":      "is",
    "Indonesian":     "id",
    "Italian":        "it",
    "Japanese":       "ja",
    "Javanese":       "jw",
    "Kannada":        "kn",
    "Kazakh":         "kk",
    "Khmer":          "km",
    "Korean":         "ko",
    "Lao":            "lo",
    "Latin":          "la",
    "Latvian":        "lv",
    "Lingala":        "ln",
    "Lithuanian":     "lt",
    "Luxembourgish":  "lb",
    "Macedonian":     "mk",
    "Malagasy":       "mg",
    "Malay":          "ms",
    "Malayalam":      "ml",
    "Maltese":        "mt",
    "Maori":          "mi",
    "Marathi":        "mr",
    "Mongolian":      "mn",
    "Nepali":         "ne",
    "Occitan":        "oc",
    "Pashto":         "ps",
    "Persian":        "fa",
    "Polish":         "pl",
    "Portuguese":     "pt",
    "Punjabi":        "pa",
    "Romanian":       "ro",
    "Russian":        "ru",
    "Sanskrit":       "sa",
    "Serbian":        "sr",
    "Shona":          "sn",
    "Sindhi":         "sd",
    "Sinhala":        "si",
    "Slovak":         "sk",
    "Slovenian":      "sl",
    "Somali":         "so",
    "Spanish":        "es",
    "Sundanese":      "su",
    "Swahili":        "sw",
    "Swedish":        "sv",
    "Tagalog":        "tl",
    "Tajik":          "tg",
    "Tamil":          "ta",
    "Tatar":          "tt",
    "Telugu":         "te",
    "Thai":           "th",
    "Tibetan":        "bo",
    "Turkish":        "tr",
    "Turkmen":        "tk",
    "Ukrainian":      "uk",
    "Urdu":           "ur",
    "Uzbek":          "uz",
    "Vietnamese":     "vi",
    "Welsh":          "cy",
    "Yiddish":        "yi",
    "Yoruba":         "yo",
}


# ── Worker (runs in background thread) ────────────────────────────────────────

class Worker(QObject):
    """Handles model loading and transcription off the main thread."""
    progress     = pyqtSignal(str)   # status bar messages
    segment_text = pyqtSignal(str)   # each transcribed segment, live
    file_done    = pyqtSignal(str)   # emitted per completed file with full output path
    finished     = pyqtSignal()
    stopped      = pyqtSignal()      # emitted when stopped mid-way by user

    def __init__(self, files, model_ref, model_size, language, cores, stop_event: threading.Event):
        super().__init__()
        self.files       = files
        self.model_ref   = model_ref
        self.model_size  = model_size
        self.language    = language
        self.cores       = cores
        self.stop_event  = stop_event

    def run(self):
        if self.stop_event.is_set():
            self.stopped.emit()
            return

        if self.model_ref[0] is None:
            self.progress.emit(f"Loading '{self.model_size}' model…")
            self.model_ref[0] = whisper_engine.load_my_model(
                model_size=self.model_size,
                cores=self.cores,
            )

        for file in self.files:
            if self.stop_event.is_set():
                self.progress.emit("Stopped.")
                self.stopped.emit()
                return

            fname = os.path.basename(file)
            self.progress.emit(f"Transcribing: {fname}")

            try:
                out_file = whisper_engine.transcribe_file(
                    file,
                    self.model_ref[0],
                    language=self.language,
                    cores=self.cores,
                    segment_callback=self._on_segment,
                    stop_event=self.stop_event,
                )
                if out_file:
                    # Return full path so the viewer can open it
                    full_out = os.path.join(os.path.dirname(file), out_file)
                    self.file_done.emit(full_out)
                else:
                    self.file_done.emit(f"⚠  Stopped mid-file: {fname}")
            except Exception as e:
                self.file_done.emit(f"✗  {fname}  ({e})")

        if not self.stop_event.is_set():
            self.progress.emit("Done!")
            self.finished.emit()

    def _on_segment(self, text: str):
        """Called by the engine for each segment; re-emits to the GUI."""
        self.segment_text.emit(text)


# ── Main Window ────────────────────────────────────────────────────────────────

class AppGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Urdu Transcriber")
        self.setMinimumSize(960, 680)

        self.selected_files: list[str] = []
        self.model_ref = [None]
        self._worker_thread: QThread | None = None
        self._stop_event = threading.Event()
        self._last_model_size: str = ""
        # Maps display label in list_right → full output path
        self._output_paths: dict[str, str] = {}

        self._build_ui()
        self._apply_styles()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 8)
        root_layout.setSpacing(10)

        # ── Row 1: file buttons ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_browse = QPushButton("📂  Browse Files")
        self.btn_browse.setFixedHeight(38)
        self.btn_browse.clicked.connect(self.browse)

        self.btn_start = QPushButton("▶  Start Processing")
        self.btn_start.setFixedHeight(38)
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.start)

        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop)

        btn_row.addWidget(self.btn_browse)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        root_layout.addLayout(btn_row)

        # ── Row 2: settings (Model | Language | Cores) ───────────────────────
        settings_row = QHBoxLayout()
        settings_row.setSpacing(16)

        settings_row.addWidget(self._make_label("Model"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["tiny", "small", "medium", "large"])
        self.combo_model.setCurrentText("small")
        self.combo_model.setFixedHeight(34)
        self.combo_model.setMinimumWidth(110)
        self.combo_model.currentTextChanged.connect(self._on_model_changed)
        settings_row.addWidget(self.combo_model)

        settings_row.addSpacing(8)

        settings_row.addWidget(self._make_label("Language"))
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(list(WHISPER_LANGUAGES.keys()))
        self.combo_lang.setCurrentText("Urdu")
        self.combo_lang.setFixedHeight(34)
        self.combo_lang.setMinimumWidth(160)
        self.combo_lang.setEditable(True)
        self.combo_lang.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        settings_row.addWidget(self.combo_lang)

        settings_row.addSpacing(8)

        settings_row.addWidget(self._make_label("Cores"))
        self.input_cores = QLineEdit("2")
        self.input_cores.setFixedHeight(34)
        self.input_cores.setFixedWidth(60)
        self.input_cores.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_cores.setValidator(QIntValidator(1, 64, self))
        self.input_cores.setObjectName("inputCores")
        settings_row.addWidget(self.input_cores)

        settings_row.addStretch()
        root_layout.addLayout(settings_row)

        # ── Divider ──────────────────────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("divider")
        root_layout.addWidget(line)

        # ── Main content area: top splitter (lists) + bottom splitter (text) ─
        #
        #   [ Input Files ]  [ Output Files ]
        #   ──────────────────────────────────
        #   [ Live / File Viewer text area   ]
        #
        outer_splitter = QSplitter(Qt.Orientation.Vertical)
        outer_splitter.setChildrenCollapsible(False)

        # Top half: two file lists side by side
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)

        header_row = QHBoxLayout()
        lbl_left  = QLabel("Input Files")
        lbl_right = QLabel("Output Files  (click to view)")
        lbl_left.setObjectName("colHeader")
        lbl_right.setObjectName("colHeader")
        header_row.addWidget(lbl_left)
        header_row.addWidget(lbl_right)
        top_layout.addLayout(header_row)

        lists_row = QHBoxLayout()
        lists_row.setSpacing(10)
        self.list_left  = QListWidget()
        self.list_right = QListWidget()
        self.list_right.setObjectName("listRight")
        self.list_right.currentItemChanged.connect(self._on_output_selected)
        lists_row.addWidget(self.list_left)
        lists_row.addWidget(self.list_right)
        top_layout.addLayout(lists_row)

        outer_splitter.addWidget(top_widget)

        # Bottom half: live text area
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 4, 0, 0)
        bottom_layout.setSpacing(4)

        text_header_row = QHBoxLayout()
        self.lbl_text_area = QLabel("Live Transcription")
        self.lbl_text_area.setObjectName("colHeader")
        self.btn_clear_text = QPushButton("Clear")
        self.btn_clear_text.setFixedHeight(24)
        self.btn_clear_text.setObjectName("btnClear")
        self.btn_clear_text.clicked.connect(self._clear_text_area)
        text_header_row.addWidget(self.lbl_text_area)
        text_header_row.addStretch()
        text_header_row.addWidget(self.btn_clear_text)
        bottom_layout.addLayout(text_header_row)

        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setObjectName("textArea")
        # Right-to-left layout for Urdu
        self.text_area.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        bottom_layout.addWidget(self.text_area)

        outer_splitter.addWidget(bottom_widget)
        outer_splitter.setSizes([300, 200])   # initial pixel heights

        root_layout.addWidget(outer_splitter, stretch=1)

        # ── Status bar ───────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    @staticmethod
    def _make_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("settingsLabel")
        return lbl

    # ── Styles ─────────────────────────────────────────────────────────────────

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: -apple-system, 'Helvetica Neue', 'Arial', sans-serif;
                font-size: 13px;
            }

            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 0 18px;
                font-size: 13px;
            }
            QPushButton:hover   { background-color: #45475a; }
            QPushButton:pressed { background-color: #585b70; }
            QPushButton:disabled { background-color: #1e1e2e; color: #45475a; border-color: #313244; }

            QPushButton#btnStart {
                background-color: #a6e3a1;
                color: #1e1e2e;
                border: none;
                font-weight: bold;
            }
            QPushButton#btnStart:hover   { background-color: #94d89a; }
            QPushButton#btnStart:pressed { background-color: #82c882; }
            QPushButton#btnStart:disabled { background-color: #45475a; color: #6c7086; }

            QPushButton#btnStop {
                background-color: #f38ba8;
                color: #1e1e2e;
                border: none;
                font-weight: bold;
            }
            QPushButton#btnStop:hover   { background-color: #eb7a97; }
            QPushButton#btnStop:pressed { background-color: #d96b86; }
            QPushButton#btnStop:disabled { background-color: #45475a; color: #6c7086; }

            QPushButton#btnClear {
                background-color: #313244;
                color: #6c7086;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 0 10px;
                font-size: 11px;
            }
            QPushButton#btnClear:hover { color: #cdd6f4; }

            QComboBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 0 10px;
            }
            QComboBox:hover { border-color: #cba6f7; }
            QComboBox::drop-down { border: none; width: 24px; }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #cdd6f4;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #45475a;
                selection-color: #cba6f7;
                border: 1px solid #45475a;
                border-radius: 4px;
            }

            QLineEdit#inputCores {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 0 6px;
                font-size: 13px;
            }
            QLineEdit#inputCores:hover { border-color: #cba6f7; }
            QLineEdit#inputCores:focus { border-color: #cba6f7; }

            QListWidget {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
                padding: 4px;
                color: #cdd6f4;
            }
            QListWidget::item { padding: 4px 6px; border-radius: 4px; }
            QListWidget::item:selected {
                background-color: #313244;
                color: #cba6f7;
            }
            QListWidget#listRight { color: #a6e3a1; }

            QTextEdit#textArea {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
                padding: 8px;
                color: #cdd6f4;
                font-size: 14px;
                line-height: 1.6;
            }

            QSplitter::handle {
                background-color: #313244;
                height: 4px;
            }
            QSplitter::handle:hover { background-color: #cba6f7; }

            QLabel#colHeader {
                color: #6c7086;
                font-size: 11px;
            }
            QLabel#settingsLabel {
                color: #6c7086;
                font-size: 12px;
            }

            QFrame#divider { color: #313244; }

            QStatusBar {
                background-color: #181825;
                color: #6c7086;
                border-top: 1px solid #313244;
                font-size: 12px;
            }
        """)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _on_model_changed(self, new_size: str):
        if new_size != self._last_model_size:
            self.model_ref[0] = None

    def _get_language(self) -> str | None:
        display = self.combo_lang.currentText().strip()
        return WHISPER_LANGUAGES.get(display, None)

    def _get_cores(self) -> int:
        text = self.input_cores.text().strip()
        return int(text) if text.isdigit() and int(text) > 0 else 2

    def _set_controls_running(self, running: bool):
        self.btn_browse.setEnabled(not running)
        self.btn_start.setEnabled(not running)
        self.combo_model.setEnabled(not running)
        self.combo_lang.setEnabled(not running)
        self.input_cores.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def _clear_text_area(self):
        self.text_area.clear()
        self.lbl_text_area.setText("Live Transcription")

    # ── Slots ──────────────────────────────────────────────────────────────────

    def browse(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio / Video Files", "",
            "Media Files (*.mp3 *.wav *.mp4 *.m4a *.ogg *.flac);;All Files (*)"
        )
        if files:
            self.selected_files = files
            self.list_left.clear()
            for f in files:
                self.list_left.addItem(os.path.basename(f))

    def start(self):
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please browse and select files first.")
            return

        model_size = self.combo_model.currentText()
        language   = self._get_language()
        cores      = self._get_cores()

        self._last_model_size = model_size
        self._stop_event.clear()
        self._output_paths.clear()

        self._set_controls_running(True)
        self.list_right.clear()
        self.text_area.clear()
        self.lbl_text_area.setText("Live Transcription")

        self._worker_thread = QThread()
        self._worker = Worker(
            self.selected_files, self.model_ref,
            model_size, language, cores,
            self._stop_event,
        )
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_bar.showMessage)
        self._worker.segment_text.connect(self._append_segment)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.finished.connect(self._on_finished)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.stopped.connect(self._worker_thread.quit)

        self._worker_thread.start()

    def stop(self):
        self._stop_event.set()
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Stopping after current segment…")

    def _append_segment(self, text: str):
        """Append a live segment to the text area and auto-scroll."""
        self.text_area.moveCursor(QTextCursor.MoveOperation.End)
        self.text_area.insertPlainText(text + " ")
        self.text_area.moveCursor(QTextCursor.MoveOperation.End)

    def _on_file_done(self, full_path: str):
        """Add completed file to right list; store path for viewer."""
        label = f"✓  {os.path.basename(full_path)}"
        self.list_right.addItem(label)
        self._output_paths[label] = full_path

    def _on_output_selected(self, current, previous):
        """Load clicked output file into the text area."""
        if current is None:
            return
        label = current.text()
        path  = self._output_paths.get(label)
        if path and os.path.exists(path):
            self.lbl_text_area.setText(f"Viewing: {os.path.basename(path)}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.text_area.setPlainText(f.read())
            except Exception as e:
                self.text_area.setPlainText(f"Could not open file:\n{e}")

    def _on_finished(self):
        self._set_controls_running(False)
        QMessageBox.information(self, "Finished", "All files processed successfully.")

    def _on_stopped(self):
        self._set_controls_running(False)
        self.status_bar.showMessage("Stopped by user.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AppGUI()
    window.show()
    sys.exit(app.exec())
