import shutil
from core.runners import EpisodeRunner
from core.registry import ProcessorRegistry
from utils.file_utils import ensure_dir
from utils.video_utils import copy_video_ffmpeg, check_video_valid

@ProcessorRegistry.register("structure")
class StructureProcessor(EpisodeRunner):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.fast_video_copy = self.config.get("fast_video_copy", False)

    def check_completed(self, context):
        """
        Check if structure step is complete.
        Requires: qpos.pt, endpose.pt, and all raw videos.
        """
        # 1. Check data files
        if not (context.dest_episode_dir / "qpos.pt").exists():
            return False
        # endpose is optional if source eef didn't exist, but usually required
        # If we can't easily know if source had eef, we skip this check or check if qpos exists is enough.
        
        # 2. Check videos
        # We need to know which cams are expected.
        # But source inspection is expensive.
        # Let's assume if raw_video dir exists and has content, it's a good sign?
        dest_raw_video_dir = context.dest_episode_dir / "raw_video"
        if not dest_raw_video_dir.exists():
            return False
        
        # If we want to be strict:
        expected_cams = ["cam_high.mp4", "cam_left_wrist.mp4", "cam_right_wrist.mp4"]
        for cam in expected_cams:
            if not (dest_raw_video_dir / cam).exists():
                return False
                
        return True

    def process_episode(self, context):
        if self.check_completed(context) and not self.config.get("overwrite", False):
            context.status = "skipped"
            context.reason = "Structure already completed"
            return context

        dest_raw_video_dir = context.dest_episode_dir / "raw_video"
        ensure_dir(dest_raw_video_dir)
        
        src_episode_dir = context.src_episode_dir
        episode_name = context.episode_name
        
        cam_map = {
            "cam_high": [f"{episode_name}_cam_high.mp4", f"{episode_name}_head.mp4", f"{episode_name}_tts.mp4", f"{episode_name}.mp4"],
            "cam_left_wrist": [f"{episode_name}_cam_left_wrist.mp4", f"{episode_name}_left_wrist.mp4"],
            "cam_right_wrist": [f"{episode_name}_cam_right_wrist.mp4", f"{episode_name}_right_wrist.mp4"],
            "cam_front": [f"{episode_name}_cam_front.mp4", f"{episode_name}_front.mp4"]
        }
        
        found_videos = {}
        for key, candidates in cam_map.items():
            for cand in candidates:
                p = src_episode_dir / cand
                if p.exists():
                    found_videos[key] = p
                    break
                p_parent = src_episode_dir.parent / cand
                if p_parent.exists():
                    found_videos[key] = p_parent
                    break

        qpos_path = src_episode_dir / f"{episode_name}_qpos.pt"
        if not qpos_path.exists():
             qpos_path = src_episode_dir.parent / f"{episode_name}_qpos.pt"
             
        # Find eef path
        eef_path = src_episode_dir / f"{episode_name}_eef.pt"
        if not eef_path.exists():
            eef_path = src_episode_dir.parent / f"{episode_name}_eef.pt"

        # Copy Data Files
        dest_qpos_path = context.dest_episode_dir / "qpos.pt"
        dest_endpose_path = context.dest_episode_dir / "endpose.pt" # Renamed from eef.pt
        
        if qpos_path.exists():
            if not dest_qpos_path.exists() or self.config.get("overwrite", False):
                shutil.copy2(qpos_path, dest_qpos_path)
            # Update context to point to the new destination file
            context.qpos_path = dest_qpos_path
        else:
            # Fallback to source if copy failed or missing, though this shouldn't happen if logic is strict
            context.qpos_path = qpos_path if qpos_path.exists() else None

        if eef_path.exists():
             if not dest_endpose_path.exists() or self.config.get("overwrite", False):
                shutil.copy2(eef_path, dest_endpose_path)

        # Copy Raw Videos
        for cam_key, src_path in found_videos.items():
            dest_path = dest_raw_video_dir / f"{cam_key}.mp4"
            
            # Check if destination exists and is valid
            should_process = True
            if dest_path.exists():
                if check_video_valid(str(dest_path)):
                    should_process = False
                else:
                    context.log(f"Found invalid video at {dest_path.name}, re-processing...")
                    dest_path.unlink()
            
            if should_process:
                if self.fast_video_copy:
                    # Direct copy
                    shutil.copy2(src_path, dest_path)
                else:
                    # FFMPEG copy (re-muxing) or re-encode
                    copy_video_ffmpeg(src_path, dest_path)
                    
            context.video_paths[cam_key] = dest_path
            
        return context
