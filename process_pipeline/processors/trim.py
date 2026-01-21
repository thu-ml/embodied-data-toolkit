import json
from core.runners import EpisodeRunner
from core.registry import ProcessorRegistry
from utils.video_utils import trim_video_ffmpeg, trim_video_cv2
from utils.data_utils import get_trim_indices, load_tensor, save_tensor

@ProcessorRegistry.register("trim")
class TrimProcessor(EpisodeRunner):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.trim_threshold = self.config.get("threshold", 0.01)
        self.trim_mode = self.config.get("video_trim_mode", "ffmpeg")
        self.trim_fps = self.config.get("video_trim_fps", 30)

    def _trim_video(self, source_path, dest_path, start_frame, end_frame):
        if self.trim_mode == "fast":
            trim_video_cv2(source_path, dest_path, start_frame, end_frame, fps=self.trim_fps)
        else:
            trim_video_ffmpeg(source_path, dest_path, start_frame, end_frame)

    def process_episode(self, context):
        qpos_path = context.qpos_path
        dest_episode_dir = context.dest_episode_dir
        dest_raw_video_dir = dest_episode_dir / "raw_video"
        dest_endpose_path = dest_episode_dir / "endpose.pt" # Should be trajectory
        
        # Load QPOS tensor (either from memory or file)
        qpos_tensor = None
        if qpos_path and qpos_path.exists():
            loaded = load_tensor(str(qpos_path))
            if isinstance(loaded, tuple):
                qpos_tensor = loaded[0]
            else:
                qpos_tensor = loaded
            
        if qpos_tensor is not None and self.trim_threshold > 0:
            ts, te = get_trim_indices(qpos_tensor, self.trim_threshold)
            
            if ts is not None:
                trim_start, trim_end = ts, te
                
                # 4a. Trim QPOS
                new_qpos = qpos_tensor[trim_start:trim_end]
                save_tensor(new_qpos, dest_episode_dir / "qpos.pt")
                
                # 4b. Trim EEF / Endpose (if exists as a trajectory)
                if dest_endpose_path.exists():
                    loaded_eef = load_tensor(str(dest_endpose_path))
                    eef_tensor = loaded_eef[0] if isinstance(loaded_eef, tuple) else loaded_eef
                    
                    if len(eef_tensor) == len(qpos_tensor):
                        new_eef = eef_tensor[trim_start:trim_end]
                        save_tensor(new_eef, dest_endpose_path)
                    else:
                        # Size mismatch, maybe eef was single frame? Keep it or update?
                        # If single frame, it likely represents the GOAL, so we keep it.
                        # But reference script implies it's a trajectory.
                        context.log(f"Warning: endpose.pt length {len(eef_tensor)} != qpos length {len(qpos_tensor)}. Skipping trim for endpose.")
                elif len(new_qpos) > 0:
                    # Fallback: if no endpose existed, use last frame of qpos?
                    # BUT StructureProcessor should have handled copying *_eef.pt to endpose.pt
                    save_tensor(new_qpos[-1], dest_endpose_path)
                   
                # 4c. Trim Videos
                ffmpeg_start = trim_start
                ffmpeg_end = trim_end - 1 
                
                # Trim raw videos
                raw_videos = [p for p in dest_raw_video_dir.glob("*.mp4") if ".temp" not in p.name]
                
                for p in raw_videos:
                    if not p.exists(): continue
                    temp_p = p.with_suffix(".temp.mp4")
                    try:
                        p.rename(temp_p)
                        self._trim_video(temp_p, p, ffmpeg_start, ffmpeg_end)
                    except Exception as e:
                        context.log(f"Error trimming video {p.name}: {e}")
                        if temp_p.exists() and not p.exists():
                            temp_p.rename(p)
                    finally:
                        if temp_p.exists():
                            temp_p.unlink()
                    
                # Trim merged video
                merged_p = dest_episode_dir / "video.mp4"
                if merged_p.exists():
                    temp_p = merged_p.with_suffix(".temp.mp4")
                    try:
                        merged_p.rename(temp_p)
                        self._trim_video(temp_p, merged_p, ffmpeg_start, ffmpeg_end)
                    except Exception as e:
                         context.log(f"Error trimming merged video: {e}")
                         if temp_p.exists() and not merged_p.exists():
                            temp_p.rename(merged_p)
                    finally:
                        if temp_p.exists():
                            temp_p.unlink()
                    
                # 4d. Update Instructions Indices
                json_path = dest_episode_dir / "instructions.json"
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            jdata = json.load(f)
                        
                        new_len = max(0, trim_end - trim_start)
                        
                        updated = False
                        if "sub_instructions" in jdata:
                            for sub in jdata["sub_instructions"]:
                                sub["start_frame"] = 0
                                sub["end_frame"] = new_len
                                updated = True
                        
                        if "total_frames" in jdata:
                            jdata["total_frames"] = new_len
                            updated = True
                            
                        if updated:
                            with open(json_path, 'w') as f:
                                json.dump(jdata, f, indent=4)
                    except Exception as e:
                        context.log(f"Error updating instructions.json: {e}")

            else:
                # No movement detected
                context.log("No movement detected above threshold. Keeping original.")
                save_tensor(qpos_tensor, dest_episode_dir / "qpos.pt")
                # Don't overwrite endpose if it exists, otherwise fallback
                if not dest_endpose_path.exists() and len(qpos_tensor) > 0:
                    save_tensor(qpos_tensor[-1], dest_endpose_path)
        else:
             if qpos_tensor is not None:
                 save_tensor(qpos_tensor, dest_episode_dir / "qpos.pt")
                 if not dest_endpose_path.exists() and len(qpos_tensor) > 0:
                     save_tensor(qpos_tensor[-1], dest_endpose_path)
                     
        return context
