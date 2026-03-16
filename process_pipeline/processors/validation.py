import numpy as np
from core.runners import EpisodeRunner
from core.registry import ProcessorRegistry
from utils.validation_utils import (
    load_tensor, 
    get_video_frame_count, 
    is_video_black_screen,
    is_video_corrupted,
    check_tensor_values_range, 
    check_tensor_static_zeros,
    check_tensor_static_repeat,
    check_tensor_group_all_zeros,
)

@ProcessorRegistry.register("validation")
class ValidationProcessor(EpisodeRunner):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.perform_validation = self.config.get("perform", True)

    def process_episode(self, context):
        if not self.perform_validation:
            return context
            
        src_episode_dir = context.src_episode_dir
        episode_name = context.episode_name
        
        # 1. Existence Checks
        qpos_path = src_episode_dir / f"{episode_name}_qpos.pt"
        if not qpos_path.exists():
            qpos_path = src_episode_dir.parent / f"{episode_name}_qpos.pt"
        
        if not qpos_path.exists():
            context.skip("Missing qpos file")
            return context
            
        eef_path = src_episode_dir / f"{episode_name}_eef.pt"
        if not eef_path.exists():
            eef_path = src_episode_dir.parent / f"{episode_name}_eef.pt"
            
        cam_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        video_paths = {}
        
        for cam in cam_names:
            cands = [
                f"{episode_name}_{cam}.mp4", 
                f"{episode_name}_{cam.replace('cam_', '')}.mp4",
            ]
            if cam == "cam_high":
                cands.append(f"{episode_name}_head.mp4")
                cands.append(f"{episode_name}_tts.mp4")
                
            found = False
            for c in cands:
                p = src_episode_dir / c
                if p.exists():
                    video_paths[cam] = p
                    found = True
                    break
                p_parent = src_episode_dir.parent / c
                if p_parent.exists():
                    video_paths[cam] = p_parent
                    found = True
                    break
            
            if not found:
                context.skip(f"Missing video for {cam}")
                return context

        # 2. Video integrity check (corrupted / unopenable)
        for cam, p in video_paths.items():
            if is_video_corrupted(str(p)):
                context.skip(f"Corrupted or unopenable video: {cam}")
                return context

        # 3. Data Loading & Frame Count Consistency
        qpos_data, qpos_len = load_tensor(str(qpos_path))
        if qpos_data is None:
            context.skip("Failed to load qpos")
            return context
            
        eef_data = None
        eef_len = None
        if eef_path.exists():
            eef_data, eef_len = load_tensor(str(eef_path))
            if eef_data is None:
                context.skip("Failed to load eef")
                return context
            if eef_len != qpos_len:
                context.skip(f"Length mismatch: qpos {qpos_len} vs eef {eef_len}")
                return context
                
        for cam, p in video_paths.items():
            v_len = get_video_frame_count(str(p))
            if v_len is None or v_len <= 0:
                context.skip(f"Failed to get frame count for {cam}")
                return context
            if v_len != qpos_len:
                context.skip(f"Length mismatch: {cam} {v_len} vs qpos {qpos_len}")
                return context
                
        # 4. Black screen check
        for cam, p in video_paths.items():
            if is_video_black_screen(str(p)):
                context.skip(f"Black screen detected in {cam}")
                return context
                
        # 5. Tensor value range check
        if not check_tensor_values_range(qpos_data, abs_threshold=10):
            context.skip("Qpos values exceed limit")
            return context
        if eef_data is not None and not check_tensor_values_range(eef_data, abs_threshold=10):
            context.skip("Eef values exceed limit")
            return context

        # 6. Group all-zeros check (first/last 7 dims entirely zero)
        if not check_tensor_group_all_zeros(qpos_data):
            context.skip("Qpos has group all-zeros (first or last 7 dims)")
            return context
        if eef_data is not None and not check_tensor_group_all_zeros(eef_data):
            context.skip("Eef has group all-zeros (first or last 7 dims)")
            return context

        # 7. Static zeros run check (100+ consecutive zeros per column)
        if not check_tensor_static_zeros(qpos_data):
            context.skip("Qpos has static zero run")
            return context
        if eef_data is not None and not check_tensor_static_zeros(eef_data):
            context.skip("Eef has static zero run")
            return context

        # 8. Static repeat check (100+ consecutive identical rows in first/last 7 dims as a group)
        if not check_tensor_static_repeat(qpos_data):
            context.skip("Qpos has 100+ consecutive identical rows (first or last 7 dims)")
            return context
        if eef_data is not None and not check_tensor_static_repeat(eef_data):
            context.skip("Eef has 100+ consecutive identical rows (first or last 7 dims)")
            return context
            
        return context
