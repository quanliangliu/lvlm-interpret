import logging
import base64
from io import BytesIO
from PIL import Image
import torch
# from torchvision.transforms.functional import to_pil_image
from transformers import LlavaForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

func_to_enable_grad = '_sample'
setattr(LlavaForConditionalGeneration, func_to_enable_grad, torch.enable_grad(getattr(LlavaForConditionalGeneration, func_to_enable_grad)))
setattr(MllamaForConditionalGeneration, func_to_enable_grad, torch.enable_grad(getattr(MllamaForConditionalGeneration, func_to_enable_grad)))

try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

def is_mllama_model(model_name_or_path):
    """Check if the model is a Llama 3.2 Vision (Mllama) model."""
    mllama_patterns = ["llama-3.2", "Llama-3.2", "mllama", "Mllama", "11B-Vision", "90B-Vision"]
    return any(pattern.lower() in model_name_or_path.lower() for pattern in mllama_patterns)

def get_processor_model(args):
    #outputs: attn_output, attn_weights, past_key_value
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # --- LLaVA-Gemma processor fix (from HF issue #41206) ---
    if "llava-gemma-2b" in args.model_name_or_path or "llava-gemma-7b" in args.model_name_or_path:
        # Force the correct values, even if they already exist but are wrong
        processor.num_additional_image_tokens = 1
        processor.patch_size = 14
        processor.vision_feature_select_strategy = "default"
    # --------------------------------------------------------

    if args.load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif args.load_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        quant_config = None

    # Determine which model class to use based on model name
    use_mllama = is_mllama_model(args.model_name_or_path)
    
    if use_mllama:
        logger.info(f"Loading Llama 3.2 Vision (Mllama) model: {args.model_name_or_path}")
        model = MllamaForConditionalGeneration.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16, 
            quantization_config=quant_config, low_cpu_mem_usage=True, device_map=args.device_map,
            attn_implementation='eager'
        )
        # Store model type for later reference
        model.is_mllama = True
        # Mllama uses a different vision model structure
        model.vision_model.config.output_attentions = True
    else:
        logger.info(f"Loading LLaVA model: {args.model_name_or_path}")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16, 
            quantization_config=quant_config, low_cpu_mem_usage=True, device_map=args.device_map,
            attn_implementation='eager'
        )
        model.is_mllama = False
        model.vision_tower.config.output_attentions = True

    # Relevancy map
    # set hooks to get attention weights
    model.enc_attn_weights = []
    #outputs: attn_output, attn_weights, past_key_value
    def forward_hook(module, inputs, output): 
        if output[1] is None:
            logger.error(
                ("Attention weights were not returned for the encoder. "
                "To enable, set output_attentions=True in the forward pass of the model. ")
            )
            return output
        
        output[1].requires_grad_(True)
        output[1].retain_grad()
        model.enc_attn_weights.append(output[1])
        return output

    hooks_pre_encoder, hooks_encoder = [], []
    # Handle different model architectures (some have .model.layers, others have .layers directly)

    language_model_layers = getattr(model.language_model, 'model', model.language_model).layers
    
    for layer in language_model_layers:
        # Mllama has both self-attention and cross-attention layers
        # Only register hooks on layers that have self_attn
        if hasattr(layer, 'self_attn'):
            hook_encoder_layer = layer.self_attn.register_forward_hook(forward_hook)
            hooks_pre_encoder.append(hook_encoder_layer)
        # Optionally, also capture cross-attention weights for Mllama
        elif hasattr(layer, 'cross_attn'):
            hook_encoder_layer = layer.cross_attn.register_forward_hook(forward_hook)
            hooks_pre_encoder.append(hook_encoder_layer)

    model.enc_attn_weights_vit = []


    def forward_hook_image_processor(module, inputs, output): 
        if output[1] is None:
            logger.warning(
                ("Attention weights were not returned for the vision model. "
                 "Relevancy maps will not be calculated for the vision model. " 
                 "To enable, set output_attentions=True in the forward pass of vision_tower. ")
            )
            return output

        output[1].requires_grad_(True)
        output[1].retain_grad()
        model.enc_attn_weights_vit.append(output[1])
        return output

    hooks_pre_encoder_vit = []
    if use_mllama:
        # Mllama has a different vision model structure
        # The vision encoder is accessed through model.vision_model
        try:
            # Try to access transformer layers in the vision model
            if hasattr(model.vision_model, 'transformer'):
                vision_layers = model.vision_model.transformer.layers
            elif hasattr(model.vision_model, 'encoder'):
                vision_layers = model.vision_model.encoder.layers
            else:
                # Fallback: try to find layers attribute
                vision_layers = []
                logger.warning("Could not find vision layers for Mllama model. Vision attention hooks not set.")
            
            for layer in vision_layers:
                if hasattr(layer, 'self_attn'):
                    hook_encoder_layer_vit = layer.self_attn.register_forward_hook(forward_hook_image_processor)
                    hooks_pre_encoder_vit.append(hook_encoder_layer_vit)
        except Exception as e:
            logger.warning(f"Could not set up vision hooks for Mllama: {e}")
    else:
        for layer in model.vision_tower.vision_model.encoder.layers:
            hook_encoder_layer_vit = layer.self_attn.register_forward_hook(forward_hook_image_processor)
            hooks_pre_encoder_vit.append(hook_encoder_layer_vit)
    
    return processor, model

def process_image(image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
    if image_process_mode == "Pad":
        def expand2square(pil_img, background_color=(122, 116, 104)):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
        image = expand2square(image)
    elif image_process_mode in ["Default", "Crop"]:
        pass
    elif image_process_mode == "Resize":
        image = image.resize((336, 336))
    else:
        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
    if max(image.size) > max_len:
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
    if return_pil:
        return image
    else:
        buffered = BytesIO()
        image.save(buffered, format=image_format)
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        return img_b64_str


def to_gradio_chatbot(state):
    ui_messages = []

    for role, msg in state.messages:
        # msg can be:
        #   - (text, image, image_process_mode) for user turns
        #   - string for assistant turns
        #   - None for a placeholder assistant while the model is generating
        if msg is None:
            # skip placeholder assistant messages created before generation
            continue

        # Build content
        if isinstance(msg, tuple):
            text, image, image_process_mode = msg

            img_html = ""
            if image is not None:
                img_b64_str = process_image(
                    image,
                    "Default",
                    return_pil=False,
                    image_format="JPEG",
                )
                img_html = (
                    f'<img src="data:image/jpeg;base64,{img_b64_str}" '
                    f'alt="user upload image" />\n'
                )

            content = img_html + text.replace("<image>", "").strip()
            ui_role = "user"   # this tuple always comes from the user
        else:
            # Assistant text (generated)
            content = msg
            ui_role = "assistant"

        ui_messages.append(
            {
                "role": ui_role,
                "content": content,
            }
        )

    return ui_messages


def move_to_device(input, device='cpu'):

    if isinstance(input, torch.Tensor):
        return input.to(device).detach()
    elif isinstance(input, list):
        return [move_to_device(inp) for inp in input]
    elif isinstance(input, tuple):
        return tuple([move_to_device(inp) for inp in input])
    elif isinstance(input, dict):
        return dict( ((k, move_to_device(v)) for k,v in input.items()))
    else:
        raise ValueError(f"Unknown data type for {input.type}")