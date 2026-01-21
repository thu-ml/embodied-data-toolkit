import subprocess
import shutil
from core.runners import EpisodeRunner
from core.registry import ProcessorRegistry
from utils.video_utils import copy_video_ffmpeg, get_config, get_video_info, concat_videos_ffmpeg, check_video_integrity

@ProcessorRegistry.register("concat")
class ConcatProcessor(EpisodeRunner):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.fps = self.config.get("fps", 30)
        self.resolution_config = self.config.get("resolution", {})

    def check_completed(self, context):
        """Check if output exists and is valid."""
        merged_video_path = context.dest_episode_dir / "video.mp4"
        if not merged_video_path.exists():
            return False
            
        # Lightweight check: size
        if merged_video_path.stat().st_size < 1024:
            return False
            
        # Optional deep check if configured or if we want to be safe
        is_valid, _ = check_video_integrity(merged_video_path)
        return is_valid

    def process_episode(self, context):
        # 1. Check if already completed (using enhanced check)
        if self.check_completed(context) and not self.config.get("overwrite", False):
            context.status = "skipped"
            context.reason = "Output video already exists and is valid"
            return context

        dest_raw_video_dir = context.dest_episode_dir / "raw_video"
        v_high = dest_raw_video_dir / "cam_high.mp4"
        v_left = dest_raw_video_dir / "cam_left_wrist.mp4"
        v_right = dest_raw_video_dir / "cam_right_wrist.mp4"
        
        merged_video_path = context.dest_episode_dir / "video.mp4"
        
        if v_high.exists() and v_left.exists() and v_right.exists():
            try:
                # Use the new shared utility function
                concat_videos_ffmpeg(
                    v_high, 
                    v_left, 
                    v_right, 
                    merged_video_path, 
                    fps=self.fps, 
                    resolution_config=self.resolution_config
                )
            except ValueError as e:
                context.fail(f"Concat resolution error: {e}")
                raise e
        elif v_high.exists():
            copy_video_ffmpeg(v_high, merged_video_path)
        else:
            context.log(f"Warning: Missing required videos for concat in {context.episode_name}")
            
        return context
