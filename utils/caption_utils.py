import os
import cv2
import json
from base64 import b64encode
from openai import OpenAI

def encode_image(image_bgr):
    """Encode OpenCV image (BGR) to base64 string."""
    _, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return b64encode(buffer).decode('utf-8')

def generate_caption_for_video(images, task_name, prefix=None, api_key=None, api_base=None):
    """
    Generate caption using OpenAI GPT-4o with a sequence of images.
    images: list of numpy arrays (BGR)
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
        
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )
    
    encoded_images = [encode_image(img) for img in images]
    model_name = "gpt-4o"
    
    system_prompt = (
        "You are a skilled robot instruction annotator. Your task is to take a given raw instruction of robot operations, "
        "and then expand the original instruction by adding details to make it detailed, rich, and accurate. "
        "You can also access images to help you build your answer, and each image is the concatenation of images from three different cameras: "
        "the camera in the center at high altitude, the camera on the left wrist, and the camera on the right wrist. "
        "Your response should be no longer than one sentence and match genuine human instructions. "
        "Try to avoid using specific numbers like 'five centimeters' and avoid modifying the position like \"forward, backward, left, right\". "
        "Try to avoid specifying which particular arm to use for the subtask. "
        "You can specify objects using colors, shapes, or other attributes. "
        "You should only respond with the expanded instruction."
    )
    
    dataset_prompt = f"The raw instruction of the task is: {task_name}"
    
    content = [{"type": "text", "text": dataset_prompt}]
    for encoded_image in encoded_images:
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
        
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": content}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=230
        )
        caption = response.choices[0].message.content
        
        if prefix:
            # Simple heuristic to prepend prefix
            caption = f"{prefix}, {caption[0].lower()}{caption[1:]}"
            
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return None

def save_instruction_json(caption, total_frames, output_path):
    """
    Save caption in the specified JSON format.
    Assumes single instruction for the whole video.
    """
    data = {
        "instructions": [caption] if caption else [],
        "sub_instructions": [
            {
                "start_frame": 0,
                "end_frame": total_frames,
                "instruction": caption if caption else ""
            }
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
