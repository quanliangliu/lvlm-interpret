import sys
sys.path = [p for p in sys.path if p is not None]

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable torch.compile

import argparse 
import logging


from utils_gradio import build_demo

logging.basicConfig(level=logging.INFO)

# Supported models:
# - LLaVA models: "Intel/llava-gemma-2b", "Intel/llava-gemma-7b", "llava-hf/llava-1.5-7b-hf", etc.
# - Llama 3.2 Vision models: "meta-llama/Llama-3.2-11B-Vision-Instruct", "meta-llama/Llama-3.2-90B-Vision-Instruct"
# - Qwen VL models: "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2.5-VL-2B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", etc.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, default="Intel/llava-gemma-2b",
                        help="Model name or path to load. Supports LLaVA models and Llama 3.2 Vision models.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server on")
    parser.add_argument("--share", action="store_true",
                        help="Whether to share the server on Gradio's public server")
    parser.add_argument("--embed", action="store_true",
                        help="Whether to run the server in an iframe")
    parser.add_argument("--load_4bit", action="store_true",
                        help="Whether to load the model in 4bit")
    parser.add_argument("--load_8bit", action="store_true",
                        help="Whether to load the model in 8bit")
    parser.add_argument("--device_map", default="auto",
                        help="Device map to use for model", choices=["auto", "cpu", "cuda", "hpu"])
    args = parser.parse_args()

    assert not( args.load_4bit and args.load_8bit), "Cannot load both 4bit and 8bit models"

    demo = build_demo(args, embed_mode=False)
    # demo.queue(max_size=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=True
    )
