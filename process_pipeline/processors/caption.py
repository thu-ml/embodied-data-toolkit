import numpy as np
from core.interface import BaseProcessor
from core.runners import EpisodeRunnerMixin, TaskRunnerMixin
from core.context import EpisodeContext, DatasetContext
from core.registry import ProcessorRegistry
from utils.video_utils import read_video_frames, get_video_frame_count
from utils.caption_utils import generate_caption_for_video, save_instruction_json

@ProcessorRegistry.register("caption")
class CaptionProcessor(BaseProcessor):
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        self.api_base = self.config.get("api_base")
        self.use_shared_caption = self.config.get("same_caption", False)
        
        # New parameters
        self.use_task_name = self.config.get("use_task_name", False)
        self.system_prompt = self.config.get("system_prompt", "")
        
        # Initialize the appropriate runner logic
        if self.use_shared_caption:
            self.runner = TaskRunnerMixin(self)
        else:
            self.runner = EpisodeRunnerMixin(self)

    def process_dataset(self, dataset_context: DatasetContext) -> DatasetContext:
        # Delegate to the chosen runner implementation
        return self.runner.process_dataset(dataset_context)

    def check_completed(self, context: EpisodeContext) -> bool:
        """Check if instructions.json exists."""
        json_path = context.dest_episode_dir / "instructions.json"
        return json_path.exists()

    def process_episode(self, context: EpisodeContext) -> EpisodeContext:
        """Called by EpisodeRunnerMixin"""
        return self._generate_and_save(context)

    def process_task(self, task_name: str, episodes: list[EpisodeContext]) -> list[EpisodeContext]:
        """Called by TaskRunnerMixin"""
        if not episodes:
            return []
            
        # 1. Generate once
        # Use the first episode to generate caption (if needed)
        # print(f"Generating shared caption for task: {task_name}") 
        # (Logging moved to reduce spam if fast)
        sample_episode = episodes[0]
        video_path = sample_episode.dest_episode_dir / "video.mp4"
        
        raw_caption = None
        
        if self.use_task_name:
            raw_caption = task_name
        elif video_path.exists():
            raw_caption = self._generate_from_api(sample_episode, video_path)
            
        if not raw_caption:
            print(f"Failed to generate caption for task {task_name}")
            return episodes

        # Prepend system prompt
        final_caption = f"{self.system_prompt}{raw_caption}"
            
        # 2. Apply to all
        for ep in episodes:
            ep.generated_caption = final_caption
            
            v_path = ep.dest_episode_dir / "video.mp4"
            if v_path.exists():
                # OPTIMIZATION: Use get_video_frame_count (ffprobe) instead of decoding all frames!
                total_frames = get_video_frame_count(str(v_path))
                save_instruction_json(final_caption, total_frames, ep.dest_episode_dir / "instructions.json")
                
        return episodes

    def _generate_and_save(self, context: EpisodeContext) -> EpisodeContext:
        video_path = context.dest_episode_dir / "video.mp4"
        if not video_path.exists():
            return context

        raw_caption = None
        if self.use_task_name:
            raw_caption = context.task_name
        else:
            raw_caption = self._generate_from_api(context, video_path)

        if raw_caption:
            final_caption = f"{self.system_prompt}{raw_caption}"
            context.generated_caption = final_caption
            # OPTIMIZATION: Use get_video_frame_count (ffprobe)
            total_frames = get_video_frame_count(str(video_path))
            save_instruction_json(final_caption, total_frames, context.dest_episode_dir / "instructions.json")
            
        return context

    def _generate_from_api(self, context, video_path):
        """Original generation logic using API."""
        frames, _ = read_video_frames(str(video_path))
        if not frames:
            return None
        num_images = 6
        indices = np.linspace(0, len(frames)-1, num_images, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        return generate_caption_for_video(
            sampled_frames, 
            context.task_name, 
            api_key=self.api_key, 
            api_base=self.api_base
        )
