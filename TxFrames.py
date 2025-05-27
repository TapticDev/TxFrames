import sys
import os
import json
import time
import threading
import multiprocessing
from functools import cache
from time import sleep
from webbrowser import open as open_browser
from subprocess import run as subprocess_run
from shutil import rmtree as remove_directory
from timeit import default_timer as timer
from typing import Callable, List, Tuple, Optional

# Third-party imports
import numpy as np
import cv2
from PIL import Image
from natsort import natsorted

#Debug
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
print("CUDA available:", 'CUDAExecutionProvider' in ort.get_available_providers())
print("DirectML available:", 'DmlExecutionProvider' in ort.get_available_providers())

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime import get_available_providers
from onnxruntime import ExecutionMode, GraphOptimizationLevel

# GUI imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QTabWidget, QFrame, QTextEdit,
    QScrollArea, QGroupBox, QLineEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QFont, QPixmap, QColor

# Initialize app constants
APP_NAME = "TxFrames"
VERSION = "1.0.0"
GITHUB_URL = "https://github.com/TapticDev/TxFrames"

# Colour scheme
COLORS = {
    "primary": "#F08080",
    "background": "#1E1E1E",
    "widget_bg": "#252526",
    "text": "#D4D4D4",
    "error": "#FF3131",
    "info": "#3399FF",
    "success": "#00AA00",
    "accent": "#007ACC"
}

# Configuration paths
DOCUMENTS_PATH = os.path.join(os.path.expanduser('~'), 'Documents')
CONFIG_PATH = os.path.join(DOCUMENTS_PATH, f"{APP_NAME}_config.json")
FFMPEG_PATH = os.path.join("extra", "ffmpeg.exe")
EXIFTOOL_PATH = os.path.join("extra", "exiftool.exe")

# Supported formats
SUPPORTED_VIDEO_EXTS = [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".gif"]
SUPPORTED_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

# AI Models
AI_MODELS = {
    "txfsmall": "Fast (Lower Quality)",
    "txflarge": "High Quality (Slower)"
}

# Frame interpolation options
INTERPOLATION_OPTIONS = {
    "2x": "2x Frames (Default)",
    "3x": "3x Frames",
    "4x": "4x Frames",
    "8x": "8x Frames",
    "16x": "16x Frames (Extreme)",
    "slow2x": "Slow Motion 2x",
    "slow4x": "Slow Motion 4x",
    "slow8x": "Slow Motion 8x"
}

# GPU options
GPU_OPTIONS = ["Auto", "GPU 0", "GPU 1", "GPU 2", "GPU 3"]

# Video codecs
VIDEO_CODECS = {
    "libx264": "H.264 (Software)",
    "libx265": "H.265/HEVC (Software)",
    "h264_nvenc": "NVIDIA H.264",
    "hevc_nvenc": "NVIDIA H.265",
    "h264_amf": "AMD H.264",
    "hevc_amf": "AMD H.265",
    "h264_qsv": "Intel H.264",
    "hevc_qsv": "Intel H.265"
}

class AppConfig:
    """Handles application configuration loading and saving"""
    def __init__(self):
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load or create configuration file"""
        defaults = {
            "ai_model": "txflarge",
            "interpolation": "2x",
            "gpu": "GPU 0",
            "video_codec": "libx264",
            "output_path": "Same as input",
            "keep_frames": False,
            "use_gpu": True
        }
        
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    return {**defaults, **json.load(f)}
            except:
                return defaults
        return defaults
    
    def save(self):
        """Save current configuration"""
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.config, f, indent=4)

class AIInterpolator:
    """Handles frame interpolation using ONNX models with GPU acceleration"""
    def __init__(self, model_name: str, gpu: str = "GPU 0", use_gpu: bool = True):
        self.model_name = model_name
        self.gpu = gpu
        self.use_gpu = use_gpu
        self.session = self._init_session()
        
    def _init_session(self) -> InferenceSession:
        """Initialize ONNX inference session with DirectML provider"""
        model_path = os.path.join("models", f"{self.model_name}_fp32.onnx")
        
        sess_options = SessionOptions()
        
        providers = []
        provider_options = []
        
        if self.use_gpu:
            match self.gpu:
                case 'Auto':
                    provider_options = [{"performance_preference": "high_performance"}]
                case 'GPU 0':
                    provider_options = [{"device_id": "0"}]
                case 'GPU 1':
                    provider_options = [{"device_id": "1"}]
                case 'GPU 2':
                    provider_options = [{"device_id": "2"}]
                case 'GPU 3':
                    provider_options = [{"device_id": "3"}]
            
            providers = ['DmlExecutionProvider']
        
        providers.append('CPUExecutionProvider')
        provider_options.append({})
        
        try:
            print(f"[GPU] Attempting to create session with providers: {providers}")
            
            session = InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options
            )
            
            active_provider = session.get_providers()[0]
            print(f"[GPU] Session created successfully with provider: {active_provider}")
            
            if 'DmlExecutionProvider' in active_provider:
                print(f"[GPU] DirectML GPU acceleration ACTIVE")
            else:
                print(f"[GPU] Using CPU execution")
                
            return session
            
        except Exception as e:
            print(f"[GPU] Session creation failed: {str(e)}")
            print("[GPU] Falling back to CPU-only execution")
            
            return InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider'],
                provider_options=[{}]
            )
        
    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, factor: int = 2) -> List[np.ndarray]:
        """Generate interpolated frames between two input frames"""
        # Normalize and concatenate input frames
        input_data = np.concatenate((frame1/255, frame2/255), axis=2).astype(np.float32)
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        output = self.session.run(None, {self.session.get_inputs()[0].name: input_data})[0]
        
        # Process output
        output = np.squeeze(output, axis=0)
        output = np.clip(output, 0, 1)
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).astype(np.uint8)
        
        frames = [output]
        
        if factor > 2:
            left_half = self.interpolate(frame1, output, factor//2)
            right_half = self.interpolate(output, frame2, factor//2)
            frames = left_half + [output] + right_half
        
        return frames

class VideoProcessor(QThread):
    """Handles video processing pipeline"""
    progress_signal = pyqtSignal(str, int)
    
    def __init__(self, config: AppConfig, input_files: list):
        super().__init__()
        self.config = config
        self.input_files = input_files
        self.ai = AIInterpolator(
            config.config["ai_model"], 
            config.config["gpu"],
            config.config["use_gpu"]
        )
        
    def run(self):
        """Process all input files"""
        try:
            for i, input_path in enumerate(self.input_files):
                self._process_video(input_path)
                progress = int((i + 1) / len(self.input_files) * 100)
                self.progress_signal.emit(f"Processed {i+1}/{len(self.input_files)} files", progress)
                
            self.progress_signal.emit("COMPLETED", 100)
            
        except Exception as e:
            self.progress_signal.emit(f"ERROR: {str(e)}", 0)
    
    def _process_video(self, input_path: str):
        """Process a single video file through the pipeline"""
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # 1. Prepare output paths
        output_dir, output_path = self._prepare_output_paths(input_path)
        self.progress_signal.emit(f"Processing: {os.path.basename(input_path)}", 0)
        
        # Get original video properties
        cap = cv2.VideoCapture(input_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Handle slow motion preprocessing if needed
        if self.config.config["interpolation"].startswith("slow"):
            # Create slowed down version first
            slowed_path = self._preprocess_slow_motion(input_path, output_dir)
            if slowed_path:
                input_path = slowed_path
        
        # 2. Extract frames
        frame_paths = self._extract_frames(input_path, output_dir)
        if not frame_paths:
            raise ValueError("No frames extracted from video")
        
        # 3. Generate interpolated frames
        self._generate_frames(frame_paths)
        
        # 4. Encode final video with correct frame rate
        self._encode_video(input_path, output_path, frame_paths, original_fps)
        
        # 5. Clean up (optional)
        if not self.config.config["keep_frames"]:
            try:
                remove_directory(output_dir)
                self.progress_signal.emit("Temporary frames cleaned up", 0)
            except Exception as e:
                self.progress_signal.emit(f"Warning: Could not clean up frames: {str(e)}", 0)
    
    def _preprocess_slow_motion(self, input_path: str, output_dir: str) -> Optional[str]:
        """Preprocess video for slow motion by slowing it down first"""
        try:
            # Extract the slow motion factor (e.g., "slow2x" -> 2)
            interp_type = self.config.config["interpolation"]
            slow_factor = int(interp_type.replace("slow", "").replace("x", ""))
            
            slowed_path = os.path.join(output_dir, "slowed_input.mp4")
            
            cmd = [
                FFMPEG_PATH,
                "-y",
                "-i", input_path,
                "-filter:v", f"setpts={slow_factor}*PTS",
                "-an",  # Remove audio
                slowed_path
            ]
            
            self.progress_signal.emit(f"Creating {slow_factor}x slowed version...", 5)
            
            result = subprocess_run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False,
                timeout=3600
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown slow motion error"
                raise RuntimeError(f"Slow motion preprocessing failed: {error_msg}")
            
            return slowed_path
            
        except Exception as e:
            self.progress_signal.emit(f"Warning: Slow motion preprocessing failed: {str(e)}", 0)
            return None
        
    def _prepare_output_paths(self, input_path: str) -> Tuple[str, str]:
        """Generate output directory and file paths"""
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        if self.config.config["output_path"] == "Same as input":
            output_dir = os.path.splitext(input_path)[0]
        else:
            output_dir = os.path.join(self.config.config["output_path"], base_name)
            
        # Add interpolation type to output filename
        interp_type = self.config.config["interpolation"]
        output_path = f"{output_dir}_txframes_{interp_type}.mp4"
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir, output_path
    
    def _extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_paths = []
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            
            if i % 2 == 0:  # Update status every 2 frames
                progress = int((i / frame_count) * 100)
                self.progress_signal.emit(f"Extracting frames: {i}/{frame_count}", progress)
                
        cap.release()
        return frame_paths
    
    def _get_interpolation_factor(self) -> Tuple[int, int]:
        """Return (total_frames, frames_to_generate) based on selected option"""
        interp = self.config.config["interpolation"]
        
        if interp.startswith("slow"):
            multiplier = int(interp.replace("slow", "").replace("x", ""))
            return (multiplier, multiplier - 1)
        else:
            multiplier = int(interp.replace("x", ""))
            return (multiplier, multiplier - 1)
    
    def _generate_frames(self, frame_paths: List[str]):
        """Generate interpolated frames using AI"""
        total_pairs = len(frame_paths) - 1
        all_frame_paths = []
        total_frames, frames_to_generate = self._get_interpolation_factor()
        
        sorted_frame_paths = natsorted(frame_paths)
        
        for i in range(total_pairs):
            all_frame_paths.append(sorted_frame_paths[i])
            
            frame1 = cv2.imread(sorted_frame_paths[i])
            frame2 = cv2.imread(sorted_frame_paths[i+1])
            
            if frame1 is None or frame2 is None:
                self.progress_signal.emit(f"ERROR: Could not load frame {i} or {i+1}", 0)
                continue
            
            try:
                if frames_to_generate > 0:
                    interpolated = self.ai.interpolate(frame1, frame2, total_frames)
                    
                    for j in range(frames_to_generate):
                        base_name = os.path.splitext(sorted_frame_paths[i])[0]
                        gen_path = f"{base_name}_interp_{j:03d}.png"
                        cv2.imwrite(gen_path, interpolated[j])
                        all_frame_paths.append(gen_path)
                        
            except Exception as e:
                self.progress_signal.emit(f"ERROR: Frame interpolation failed: {str(e)}", 0)
                continue
                
            if i % 1 == 0:  # Update status every frame
                progress = int((i / total_pairs) * 100)
                self.progress_signal.emit(f"Generating frames: {progress}% ({i}/{total_pairs})", progress)
        
        all_frame_paths.append(sorted_frame_paths[-1])
        
        frame_paths.clear()
        frame_paths.extend(all_frame_paths)
    
    def _encode_video(self, input_path: str, output_path: str, frame_paths: List[str], original_fps: float):
        """Encode final video with FFmpeg with correct frame rate"""
        # Check if FFmpeg exists
        if not os.path.exists(FFMPEG_PATH):
            raise FileNotFoundError("FFmpeg not found. Please ensure ffmpeg.exe is in the Assets folder.")
        
        # Calculate output frame rate
        interp = self.config.config["interpolation"]
        if interp.startswith("slow"):
            output_fps = original_fps
        else:
            total_frames, _ = self._get_interpolation_factor() 
            output_fps = original_fps * total_frames
        
        temp_dir = os.path.dirname(output_path)
        list_file = os.path.join(temp_dir, "frame_list.txt")
        
        try:
            with open(list_file, "w") as f:
                for path in natsorted(frame_paths):
                    abs_path = os.path.abspath(path).replace("\\", "/")
                    f.write(f"file '{abs_path}'\n")
            
            # Build FFmpeg command
            abs_ffmpeg = os.path.abspath(FFMPEG_PATH)
            abs_list_file = os.path.abspath(list_file)
            abs_output = os.path.abspath(output_path)
            
            # Base command
            cmd = [
                abs_ffmpeg,
                "-y",  # Overwrite output
                "-hide_banner",
                "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-r", str(output_fps),
                "-i", abs_list_file,
            ]
            
            # Add audio unless it's slow motion
            if not interp.startswith("slow"):
                cmd.extend([
                    "-i", input_path,
                    "-map", "0:v",  # Video from first input (frames)
                    "-map", "1:a?",  # Audio from second input (optional)
                    "-shortest",     # End when the shortest stream ends
                ])
            else:
                cmd.extend([
                    "-map", "0:v",   # Video only
                ])
            
            # Add video encoding parameters
            cmd.extend([
                "-c:v", self.config.config["video_codec"],
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "18",
                abs_output
            ])
            
            # Run FFmpeg
            self.progress_signal.emit(f"Encoding video at {output_fps:.2f} FPS with {self.config.config['video_codec']}...", 90)
            
            result = subprocess_run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown encoding error"
                raise RuntimeError(f"Video encoding failed: {error_msg}")
            else:
                self.progress_signal.emit("Video encoding completed successfully", 95)
                
        except Exception as e:
            raise RuntimeError(f"Video encoding failed: {str(e)}")
        finally:
            if os.path.exists(list_file):
                try:
                    os.remove(list_file)
                except:
                    pass

class TxFramesApp(QMainWindow):
    """Main application window with modern design"""
    def __init__(self):
        super().__init__()
        self.config = AppConfig()
        self.selected_files = []
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the main UI components"""
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.setMinimumSize(900, 700)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create header
        self._create_header(main_layout)
        
        # Create file selection area
        self._create_file_panel(main_layout)
        
        # Create settings tabs
        self._create_settings_tabs(main_layout)
        
        # Create status area
        self._create_status_panel(main_layout)
        
        # Apply styles
        self._apply_styles()
        
    def _create_header(self, parent_layout):
        """Create application header"""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # App title
        title = QLabel(f"{APP_NAME} {VERSION}")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['accent']};")
        header_layout.addWidget(title)
        
        # Spacer
        header_layout.addStretch()
        
        # GitHub button
        github_btn = QPushButton("GitHub")
        github_btn.setFixedSize(80, 30)
        github_btn.clicked.connect(lambda: open_browser(GITHUB_URL))
        header_layout.addWidget(github_btn)
        
        parent_layout.addWidget(header)
    
    def _create_file_panel(self, parent_layout):
        """Create file selection panel"""
        group = QGroupBox("Input Files")
        group.setStyleSheet("QGroupBox { border: 1px solid #3A3A3A; border-radius: 5px; }")
        layout = QVBoxLayout(group)
        
        # Drop area
        drop_area = QLabel("Drag & Drop Video Files Here")
        drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_area.setStyleSheet("""
            QLabel {
                background-color: #2D2D30;
                border: 2px dashed #3A3A3A;
                border-radius: 5px;
                color: #D4D4D4;
                font-size: 14px;
                padding: 40px;
            }
        """)
        drop_area.setMinimumHeight(150)
        layout.addWidget(drop_area)
        
        # Or select button
        select_btn = QPushButton("OR SELECT FILES")
        select_btn.setFixedSize(140, 35)
        select_btn.clicked.connect(self._select_files)
        layout.addWidget(select_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Selected files label
        self.selected_files_label = QLabel("No files selected")
        self.selected_files_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.selected_files_label)
        
        parent_layout.addWidget(group)
    
    def _create_settings_tabs(self, parent_layout):
        """Create settings tabs"""
        tab_widget = QTabWidget()
        
        # AI Settings Tab
        ai_tab = self._create_ai_settings_tab()
        tab_widget.addTab(ai_tab, "AI Settings")
        
        # Output Settings Tab
        output_tab = self._create_output_settings_tab()
        tab_widget.addTab(output_tab, "Output Settings")
        
        # Advanced Tab
        advanced_tab = self._create_advanced_tab()
        tab_widget.addTab(advanced_tab, "Advanced")
        
        parent_layout.addWidget(tab_widget)
        
        # Process button
        self.process_btn = QPushButton("START PROCESSING")
        self.process_btn.setFixedHeight(40)
        self.process_btn.clicked.connect(self._process_files)
        parent_layout.addWidget(self.process_btn)
    
    def _create_ai_settings_tab(self):
        """Create AI settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # AI Model selection
        self._create_labeled_dropdown(
            tab, layout, "AI Model:", 
            list(AI_MODELS.keys()), 
            self.config.config["ai_model"], 
            self._update_ai_model
        )
        
        # Interpolation type
        self._create_labeled_dropdown(
            tab, layout, "Interpolation:",
            list(INTERPOLATION_OPTIONS.keys()),
            self.config.config["interpolation"],
            self._update_interpolation
        )
        
        # GPU selection
        self._create_labeled_dropdown(
            tab, layout, "GPU Device:",
            GPU_OPTIONS,
            self.config.config["gpu"],
            self._update_gpu
        )
        
        # Use GPU checkbox
        self.use_gpu_check = QCheckBox("Enable GPU Acceleration")
        self.use_gpu_check.setChecked(self.config.config["use_gpu"])
        self.use_gpu_check.stateChanged.connect(self._toggle_gpu)
        layout.addWidget(self.use_gpu_check, alignment=Qt.AlignmentFlag.AlignCenter)
        
        return tab
    
    def _create_output_settings_tab(self):
        """Create output settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Output path
        self._create_output_path_selector(tab, layout)
        
        # Video codec
        self._create_labeled_dropdown(
            tab, layout, "Video Codec:",
            list(VIDEO_CODECS.keys()),
            self.config.config["video_codec"],
            self._update_codec
        )
        
        # Keep frames checkbox
        self.keep_frames_check = QCheckBox("Keep Temporary Frames")
        self.keep_frames_check.setChecked(self.config.config["keep_frames"])
        self.keep_frames_check.stateChanged.connect(self._toggle_keep_frames)
        layout.addWidget(self.keep_frames_check, alignment=Qt.AlignmentFlag.AlignCenter)
        
        return tab
    
    def _create_advanced_tab(self):
        """Create advanced settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Placeholder
        label = QLabel("Advanced Settings Coming Soon")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        
        return tab
    
    def _create_labeled_dropdown(self, parent, layout, label, options, default, callback):
        """Helper to create labeled dropdown"""
        frame = QWidget(parent)
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        lbl = QLabel(label)
        lbl.setFixedWidth(120)
        frame_layout.addWidget(lbl)
        
        # Dropdown
        combo = QComboBox()
        combo.addItems(options)
        combo.setCurrentText(default)
        combo.currentTextChanged.connect(callback)
        combo.setFixedWidth(250)
        frame_layout.addWidget(combo)
        
        layout.addWidget(frame)
    
    def _create_output_path_selector(self, parent, layout):
        """Create output path selector"""
        frame = QWidget(parent)
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        lbl = QLabel("Output Path:")
        lbl.setFixedWidth(120)
        frame_layout.addWidget(lbl)
        
        # Entry
        self.output_path_entry = QLineEdit()
        self.output_path_entry.setText(self.config.config["output_path"])
        self.output_path_entry.setReadOnly(True)
        self.output_path_entry.setFixedWidth(180)
        frame_layout.addWidget(self.output_path_entry)
        
        # Browse button
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._select_output_path)
        frame_layout.addWidget(browse_btn)
        
        layout.addWidget(frame)
    
    def _create_status_panel(self, parent_layout):
        """Create status display with progress bar"""
        frame = QWidget()
        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        
        # Status label
        self.status_label = QLabel("Ready")
        frame_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setValue(0)
        frame_layout.addWidget(self.progress_bar)
        
        parent_layout.addWidget(frame)
    
    def _apply_styles(self):
        """Apply styles to the application"""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
                color: {COLORS['text']};
            }}
            
            QGroupBox {{
                color: {COLORS['text']};
                font-size: 12px;
                margin-top: 10px;
            }}
            
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
            }}
            
            QPushButton:hover {{
                background-color: #0062A3;
            }}
            
            QPushButton:disabled {{
                background-color: #505050;
            }}
            
            QComboBox, QLineEdit {{
                background-color: {COLORS['widget_bg']};
                color: {COLORS['text']};
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 5px;
            }}
            
            QProgressBar {{
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                text-align: center;
            }}
            
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                width: 10px;
            }}
        """)
    
    def _select_files(self):
        """Handle file selection with validation"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Video files (*.mp4 *.mkv *.avi *.mov *.webm *.flv *.gif)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        
        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            
            valid_files = []
            for f in files:
                if os.path.exists(f) and os.path.splitext(f)[1].lower() in SUPPORTED_VIDEO_EXTS:
                    valid_files.append(f)
            
            if valid_files:
                self.selected_files = valid_files
                self.selected_files_label.setText(f"{len(valid_files)} file(s) selected")
                self.selected_files_label.setStyleSheet(f"color: {COLORS['success']};")
            else:
                self.selected_files_label.setText("No valid video files selected")
                self.selected_files_label.setStyleSheet(f"color: {COLORS['error']};")
        else:
            self.selected_files_label.setText("No files selected")
            self.selected_files_label.setStyleSheet(f"color: {COLORS['text']};")
    
    def _select_output_path(self):
        """Handle output path selection"""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_path_entry.setText(path)
            self.config.config["output_path"] = path
    
    def _update_ai_model(self, value):
        self.config.config["ai_model"] = value
    
    def _update_interpolation(self, value):
        self.config.config["interpolation"] = value
    
    def _update_gpu(self, value):
        self.config.config["gpu"] = value
    
    def _update_codec(self, value):
        self.config.config["video_codec"] = value
    
    def _toggle_gpu(self, state):
        self.config.config["use_gpu"] = state == 2 
    
    def _toggle_keep_frames(self, state):
        self.config.config["keep_frames"] = state == 2
    
    def _process_files(self):
        """Start processing selected files"""
        if not self.selected_files:
            self.status_label.setText("No files selected")
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")
            return
            
        self.process_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        self.status_label.setStyleSheet(f"color: {COLORS['info']};")
        self.progress_bar.setValue(0)
        
        # Save current config
        self.config.save()
        
        # Start processing in background thread
        self.processor = VideoProcessor(self.config, self.selected_files)
        self.processor.progress_signal.connect(self._update_progress)
        self.processor.finished.connect(self._processing_finished)
        self.processor.start()
    
    def _update_progress(self, status, progress):
        """Update progress from worker thread"""
        self.status_label.setText(status)
        self.progress_bar.setValue(progress)
        
        if "ERROR" in status:
            self.status_label.setStyleSheet(f"color: {COLORS['error']};")
        elif "COMPLETED" in status:
            self.status_label.setStyleSheet(f"color: {COLORS['success']};")
    
    def _processing_finished(self):
        """Called when processing is complete"""
        self.process_btn.setEnabled(True)
        
        # Show completion message if no errors
        if "ERROR" not in self.status_label.text():
            QMessageBox.information(self, "Processing Complete", "All files processed successfully!")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = TxFramesApp()
    window.show()
    sys.exit(app.exec())