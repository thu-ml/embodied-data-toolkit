import os
import sys
import torch
import json
import multiprocessing
from pathlib import Path
from core.runners import EpisodeRunner
from core.context import EpisodeContext
from core.registry import ProcessorRegistry

# Global cache for the model to avoid reloading in sequential processing
_WAN_MODEL_CACHE = None
_WAN_MODEL_DEVICE = None

@ProcessorRegistry.register("t5_wan")
class T5WanProcessor(EpisodeRunner):
    def __init__(self, config=None):
        super().__init__(config)
        self.wan_dir = self.config.get("wan_dir", "/share/home/lht/cosmos-predict2/Wan2.2")
        self.cache_dir = self.config.get("cache_dir", "Wan2.2-TI2V-5B")
        self.max_length = self.config.get("max_length", 512)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Inject Wan2.2 path
        if self.wan_dir not in sys.path:
            sys.path.insert(0, self.wan_dir)

        # Auto-adjust workers for GPU to avoid contention/OOM
        if "cuda" in self.device and torch.cuda.is_available():
            try:
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    if self.workers > num_gpus:
                        print(f"[T5WanProcessor] Reducing workers from {self.workers} to {num_gpus} (Available GPUs)")
                        self.workers = num_gpus
            except Exception as e:
                print(f"[T5WanProcessor] Warning: Failed to check GPU count: {e}")

    def _get_model(self):
        global _WAN_MODEL_CACHE, _WAN_MODEL_DEVICE
        if _WAN_MODEL_CACHE is None:
            try:
                # Lazy import
                from wan.modules.t5 import T5EncoderModel
                
                # Dynamic device selection for multi-process distribution
                device_to_use = self.device
                if device_to_use == "cuda" and torch.cuda.is_available():
                    try:
                        num_gpus = torch.cuda.device_count()
                        if num_gpus > 0:
                            # Use PID modulo GPU count for distribution
                            idx = os.getpid() % num_gpus
                            device_to_use = f"cuda:{idx}"
                    except Exception:
                        pass

                print(f"[T5WanProcessor] Loading T5 Model from {self.cache_dir} on {device_to_use}...")
                
                _WAN_MODEL_CACHE = T5EncoderModel(
                    text_len=self.max_length,
                    dtype=torch.bfloat16,
                    device=device_to_use,
                    checkpoint_path=os.path.join(self.cache_dir, 'models_t5_umt5-xxl-enc-bf16.pth'),
                    tokenizer_path=os.path.join(self.cache_dir, 'google/umt5-xxl'),
                )
                _WAN_MODEL_DEVICE = device_to_use
                print("[T5WanProcessor] Model loaded.")
            except ImportError as e:
                print(f"Failed to import wan.modules.t5. Check wan_dir: {self.wan_dir}. Error: {e}")
                raise ImportError(f"Failed to import wan.modules.t5. Check wan_dir: {self.wan_dir}. Error: {e}")
            except Exception as e:
                print(f"Failed to load T5 model: {e}")
                raise RuntimeError(f"Failed to load T5 model: {e}")
        return _WAN_MODEL_CACHE, _WAN_MODEL_DEVICE

    def check_completed(self, context: EpisodeContext) -> bool:
        """
        Check if embedding files actually exist and are not empty.
        Simply checking folder existence is not enough.
        """
        json_path = context.dest_episode_dir / "instructions.json"
        if not json_path.exists():
            return False
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            sub_instructions = data.get("sub_instructions", [])
            if not sub_instructions:
                return True
                
            # Check if all instructions have embedding_path and file exists
            umt5_dir = context.dest_episode_dir / "umt5_wan"
            if not umt5_dir.exists():
                return False
                
            for item in sub_instructions:
                text = item.get("instruction", "")
                if not text:
                    continue
                    
                emb_path = item.get("embedding_path")
                if not emb_path:
                    return False
                    
                full_emb_path = context.dest_episode_dir / emb_path
                if not full_emb_path.exists() or full_emb_path.stat().st_size == 0:
                    return False
            
            return True
            
        except Exception:
            return False

    def process_episode(self, context: EpisodeContext) -> EpisodeContext:
        
        episode_path = context.dest_episode_dir
        json_path = episode_path / "instructions.json"
        
        if not json_path.exists():
            context.log("[T5WanProcessor] Skipped: No instructions.json found")
            return context

        try:
            # Load instructions
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if processing is needed
            sub_instructions = data.get("sub_instructions", [])
            if not sub_instructions:
                context.log("[T5WanProcessor] Skipped: No sub_instructions to process")
                return context

            # Ensure output directory
            output_dir = episode_path / "umt5_wan"
            output_dir.mkdir(exist_ok=True)

            model, device = self._get_model()
            updated = False

            for idx, item in enumerate(sub_instructions):
                text_list = item.get("instruction", [])
                # Normalize to string if list
                if isinstance(text_list, list):
                    text = " ".join(text_list).strip()
                else:
                    text = str(text_list).strip()
                    
                if not text:
                    continue
                
                # Generate filename hash or simple index
                save_name = f"instruction_{idx}.pt"
                save_path = output_dir / save_name
                
                # Encode
                with torch.no_grad():
                    # Pass the correct device to model call
                    res = model([text], device)
                    if isinstance(res, (tuple, list)):
                        encoded_text = res[0]
                    else:
                        encoded_text = res
                
                if isinstance(encoded_text, torch.Tensor):
                    encoded_tensor = encoded_text.cpu()
                else:
                    encoded_tensor = torch.from_numpy(encoded_text)
                
                torch.save(encoded_tensor, save_path)
                
                # Update JSON with relative path
                item["embedding_path"] = f"umt5_wan/{save_name}"
                updated = True

            if updated:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                context.log(f"[T5WanProcessor] Success: Generated embeddings for {len(sub_instructions)} instructions")
            else:
                context.log("[T5WanProcessor] Skipped: No valid instructions to process")
                
            return context

        except Exception as e:
            context.log(f"[T5WanProcessor] Error: {str(e)}")
            context.fail(str(e))
            return context

T5WANProcessor = T5WanProcessor
