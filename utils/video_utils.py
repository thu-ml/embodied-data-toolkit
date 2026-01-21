import cv2
import numpy as np
import ffmpeg
import subprocess
import os
import shutil

class FFmpegConfig:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FFmpegConfig, cls).__new__(cls)
            cls._instance._detect_config()
        return cls._instance
    
    def _detect_config(self):
        self.ffmpeg_bin = None
        self.ffprobe_bin = None
        
        # 1. Look for ffmpeg and ffprobe binaries
        # Prioritize environment variables
        if 'FFMPEG_BINARY' in os.environ:
             self.ffmpeg_bin = os.environ['FFMPEG_BINARY']
        
        # Look for ffmpeg
        candidates = ['ffmpeg', '/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg']
        if self.ffmpeg_bin:
            candidates.insert(0, self.ffmpeg_bin)
            
        for bin_path in candidates:
            if shutil.which(bin_path):
                 self.ffmpeg_bin = bin_path
                 break
        
        if self.ffmpeg_bin is None:
            self.ffmpeg_bin = 'ffmpeg'

        # Look for ffprobe (usually in the same directory as ffmpeg)
        if os.path.isabs(self.ffmpeg_bin):
            dir_name = os.path.dirname(self.ffmpeg_bin)
            probe_candidate = os.path.join(dir_name, 'ffprobe')
            if shutil.which(probe_candidate):
                self.ffprobe_bin = probe_candidate
        
        if self.ffprobe_bin is None:
            if shutil.which('ffprobe'):
                self.ffprobe_bin = 'ffprobe'
            else:
                self.ffprobe_bin = 'ffprobe'

        # 2. Encoder configuration - Align with reference code, enforce high quality
        # Reference logic from prepare_mp4_data_spatial_diverse.py
        self.encoder = 'libx264'
        self.encoder_args = ['-preset', 'slow', '-crf', '10'] # veryslow
        
        print(f"[FFmpegConfig] FFmpeg: {self.ffmpeg_bin}, FFprobe: {self.ffprobe_bin}")
        print(f"[FFmpegConfig] Encoder: {self.encoder}, Args: {self.encoder_args}")

    @property
    def cmd_base(self):
        return [self.ffmpeg_bin]

def get_config():
    return FFmpegConfig()

def check_video_valid(file_path):
    """Quickly check if video is valid using ffprobe."""
    config = get_config()
    try:
        ffmpeg.probe(file_path, cmd=config.ffprobe_bin)
        return True
    except:
        return False

def check_video_integrity(file_path):
    """
    Deep check if video is corrupted by decoding it.
    Equivalent to: ffmpeg -v error -i file -f null -
    Returns: (is_valid: bool, error_message: str)
    """
    config = get_config()
    try:
        cmd = config.cmd_base + [
            '-v', 'error',
            '-i', str(file_path),
            '-f', 'null',
            '-'
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0 or result.stderr:
            return False, result.stderr.decode("utf-8", errors="ignore")
        return True, ""
    except Exception as e:
        return False, str(e)

def get_video_info(file_path):
    """Get video metadata using ffmpeg probe."""
    config = get_config()
    try:
        # Use detected ffprobe path
        probe = ffmpeg.probe(file_path, cmd=config.ffprobe_bin)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            try:
                fps_str = video_stream.get('r_frame_rate', '30/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den != 0 else 30
                else:
                    fps = float(fps_str)
            except Exception:
                fps = 30.0
                
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Also get number of frames if available
            nb_frames = video_stream.get('nb_frames')
            if nb_frames:
                nb_frames = int(nb_frames)
            else:
                # If not available, might need to estimate or scan, but for now return None or estimate
                pass
                
            return fps, width, height, nb_frames
    except Exception as e:
        # If specified path fails, try default path (fallback)
        # Note: We catch generic Exception because in some environments ffmpeg.Error might not be available
        error_msg = str(e)
        if hasattr(e, 'stderr') and e.stderr:
             error_msg = e.stderr.decode('utf8') if isinstance(e.stderr, bytes) else e.stderr
             
        print(f"Warning: Probe with {config.ffprobe_bin} failed, trying default. Error: {error_msg}")
        try:
            probe = ffmpeg.probe(file_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream:
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                nb_frames = int(video_stream.get('nb_frames', 0))
                return 30.0, width, height, nb_frames
        except Exception as e2:
            print(f"Error probing {file_path}: {e2}")
             
    return 30.0, 640, 480, 0

def get_video_frame_count(file_path):
    """
    Get exact frame count using ffprobe. 
    First tries fast probe (nb_frames), then falls back to count packets if needed.
    """
    config = get_config()
    try:
        # Fast way: check container metadata
        fps, w, h, nb_frames = get_video_info(file_path)
        if nb_frames and nb_frames > 0:
            return nb_frames
            
        # Slow way: count packets (still faster than decoding frames)
        # ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 input.mp4
        cmd = [
            config.ffprobe_bin,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0',
            str(file_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
            
        # Fallback to slower decoding count if packet count fails (rare)
        return 0
    except Exception:
        return 0

def read_video_frames(file_name, width=640, height=480, resize_width=None, resize_height=None):
    """
    Read video frames using ffmpeg pipe. 
    Returns list of frames (numpy arrays) and fps.
    """
    config = get_config()
    fps, orig_w, orig_h, _ = get_video_info(file_name)
    
    # Use the probed resolution if width/height not provided or generic
    read_w = width if width else orig_w
    read_h = height if height else orig_h
    
    process = (
        ffmpeg
        .input(file_name)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', threads=1, v='fatal')
        .run_async(pipe_stdout=True, cmd=config.ffmpeg_bin)
    )
    
    frames = []
    frame_size = orig_w * orig_h * 3
    
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break
        if len(in_bytes) != frame_size:
            break
            
        image = np.frombuffer(in_bytes, np.uint8).reshape([orig_h, orig_w, 3])
        
        if resize_width and resize_height:
            image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        elif width != orig_w or height != orig_h:
            pass
            
        frames.append(image)
        
    process.stdout.close()
    process.wait()
    return frames, fps

def write_video(images, save_path, fps=30, width=640, height=720):
    """Write list of images to video file using ffmpeg."""
    if not images:
        return
    
    config = get_config()
    
    # Construct command
    ffmpeg_cmd = config.cmd_base + [
        '-y',                       # Overwrite output file
        '-v', 'fatal',              # Log level
        '-f', 'rawvideo',           # Input format is raw video
        '-vcodec', 'rawvideo',      # Input codec is raw video
        '-threads', '1',            # Threads
        '-s', f'{width}x{height}',  # Resolution
        '-pix_fmt', 'bgr24',        # OpenCV default pixel format
        '-r', str(fps),             # FPS
        '-i', '-',                  # Read from stdin
        '-c:v', config.encoder,     # Video codec
    ]
    
    ffmpeg_cmd += config.encoder_args
    ffmpeg_cmd += [
        '-pix_fmt', 'yuv420p',      # Compatibility pixel format
        str(save_path)
    ]
    
    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        for image in images:
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        proc.wait()
    except Exception as e:
        print(f'Error writing {save_path}: {e}')

def concat_videos_ffmpeg(v_high, v_left, v_right, output_path, fps=30, resolution_config=None):
    """
    Use ffmpeg filter_complex to stack videos:
    Top: High
    Bottom Left: Left
    Bottom Right: Right
    """
    config = get_config()
    resolution_config = resolution_config or {}
    
    # 1. Determine Target Resolutions
    if "main" in resolution_config and "wrist" in resolution_config:
        # Case A: Config driven
        main_w, main_h = resolution_config["main"]
        wrist_w, wrist_h = resolution_config["wrist"]
    else:
        # Case B: Input driven (Dynamic Probe)
        # Just probe main video to set target resolution
        # We enforce wrist videos to be resized to half of main video
        _, hw, hh, _ = get_video_info(str(v_high))
        
        main_w, main_h = hw, hh
        wrist_w, wrist_h = hw // 2, hh // 2

    # 2. Construct filter complex string
    # [0:v] is v_high, [1:v] is v_left, [2:v] is v_right
    # Force scaling of wrist videos to target dimension
    
    filter_str = (
        f"[0:v]scale={main_w}:{main_h}[top];"
        f"[1:v]scale={wrist_w}:{wrist_h}[bl];"
        f"[2:v]scale={wrist_w}:{wrist_h}[br];"
        f"[bl][br]hstack=inputs=2[bot];"
        f"[top][bot]vstack=inputs=2[out]"
    )

    cmd = config.cmd_base + [
        '-i', str(v_high),
        '-i', str(v_left),
        '-i', str(v_right),
        '-filter_complex', filter_str,
        '-map', '[out]',
        '-c:v', config.encoder, # libx264
    ]
    
    # Add encoder args (preset veryslow, crf 10 from config)
    cmd += config.encoder_args
    
    # Output options
    cmd += [
        '-r', str(fps),
        '-pix_fmt', 'yuv420p', # Ensure compatibility
        '-an', # Remove audio
        '-y',  # Overwrite
        '-v', 'fatal', # Quiet
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True)

def trim_video_ffmpeg(source_path, dest_path, start_frame, end_frame):
    """
    Trim video using ffmpeg select filter.
    Indices are inclusive.
    """
    config = get_config()
    
    ffmpeg_cmd = config.cmd_base + [
        '-i', str(source_path),
        '-vf', f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS",
        '-c:v', config.encoder, # Force re-encoding to fix timestamp/format issues
    ]
    # Add encoding params
    ffmpeg_cmd += config.encoder_args
    
    ffmpeg_cmd += [
        '-an',
        '-v', 'fatal',
        '-y',
        str(dest_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True)

def trim_video_cv2(source_path, dest_path, start_frame, end_frame, fps=30):
    """Trim video using OpenCV read/write (faster but lower compression efficiency)."""
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise Exception(f"Could not open video {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(dest_path), fourcc, orig_fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

def copy_video_ffmpeg(source_path, dest_path):
    """Simple copy/transcode."""
    config = get_config()
    
    # Change to re-encode instead of copy
    # If source video has issues (e.g. VFR, encoding anomalies), copy will preserve them
    # Force transcoding can normalize output
    ffmpeg_cmd = config.cmd_base + [
        '-i', str(source_path),
        '-c:v', config.encoder, # Use high quality encoder from config
    ]
    ffmpeg_cmd += config.encoder_args
    ffmpeg_cmd += [
        '-an', # Remove audio (if any)
        '-y',
        '-v', 'fatal',
        str(dest_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True)
