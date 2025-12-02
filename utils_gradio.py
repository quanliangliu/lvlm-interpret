import os
import tempfile
import logging

import torch

from PIL import Image
import numpy as np
import gradio as gr
import spaces

from torchvision.transforms.functional import to_pil_image

from PIL import Image as PILImage

from utils_model import get_processor_model, move_to_device, to_gradio_chatbot, process_image

from utils_attn import (
    handle_attentions_i2t, plot_attention_analysis, handle_relevancy, handle_text_relevancy, reset_tokens,
    plot_text_to_image_analysis, handle_box_reset, boxes_click_handler, attn_update_slider
)

from utils_relevancy import construct_relevancy_map

from utils_causal_discovery import (
    handle_causality, handle_causal_head, causality_update_dropdown
)

logger = logging.getLogger(__name__)

N_LAYERS = 40  # Default for Llama 3.2 11B Vision (can be overridden)
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROLE0 = "USER"
ROLE1 = "ASSISTANT"

# Image token constants for different model types
LLAVA_IMAGE_TOKEN = "<image>"
MLLAMA_IMAGE_TOKEN = "<|image|>"
QWEN_VL_IMAGE_TOKEN = "<|image_pad|>"  # Qwen VL uses <|image_pad|> as placeholder, but we use <|vision_start|>...<|vision_end|> in prompt

processor = None
model = None

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
# system_prompt = ""
# system_prompt ="""A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."""

title_markdown = ("""
# LVLM-Interpret: An Interpretability Tool for Large Vision-Language Models
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
""")

block_css = """

#image_canvas canvas {
    max-width: 400px !important;
    max-height: 400px !important;
}

#buttons button {
    min-width: min(120px,100%);
}

"""

def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = gr.State()
    state.messages = []
    return (state, [], "", None, None, None, None)

def add_text(state, text, image, image_process_mode):
    global processor
    global model

    # (You currently always reset state; keeping your behavior)
    if True:  # state is None:
        state = gr.State()
        state.messages = []

    # Handle ImageEditor dict and blank canvas
    if isinstance(image, dict):
        image = image.get("composite", None)
        if image is not None:
            background = Image.new("RGBA", image.size, (255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")

            # ImageEditor does not return None image; treat pure white as "no image"
            if (np.array(image) == 255).all():
                image = None

    # Safely trim text
    text = (text or "")[:1536]  # Hard cut-off
    logger.info(text)

    prompt_len = 0

    # Determine image token based on model type
    is_mllama = getattr(model, 'is_mllama', False) if model is not None else False
    is_qwen_vl = getattr(model, 'is_qwen_vl', False) if model is not None else False
    
    if is_qwen_vl:
        # Qwen VL uses a special format with vision tags - processor handles image insertion
        image_token = ""  # Will be handled by processor
    elif is_mllama:
        image_token = MLLAMA_IMAGE_TOKEN
    else:
        image_token = LLAVA_IMAGE_TOKEN

    # Build prompt depending on whether chat_template exists and whether we have an image
    if processor.tokenizer.chat_template is not None:
        if image is not None:
            if is_qwen_vl:
                # For Qwen VL, use special content format that processor understands
                # The processor expects a list of content items for multimodal input
                user_content = [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
                # Use processor's apply_chat_template which handles the image placeholder
                prompt = processor.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                user_content = f"{image_token}\n" + text
                prompt = processor.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            user_content = text
            prompt = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_len += len(prompt)
    else:
        prompt = system_prompt
        prompt_len += len(prompt)

        if image is not None:
            msg = f"\n{ROLE0}: {image_token}\n{text}\n{ROLE1}:"
        else:
            msg = f"\n{ROLE0}: {text}\n{ROLE1}: "
        prompt += msg
        prompt_len += len(msg)

    # Store message for UI / history
    if image is not None:
        # Keep tuple only when we actually have an image
        state.messages.append([ROLE0, (text, image, image_process_mode)])
    else:
        # Text-only turn
        state.messages.append([ROLE0, text])

    # Placeholder assistant message to be filled by lvlm_bot
    state.messages.append([ROLE1, None])

    state.prompt_len = prompt_len
    state.prompt = prompt

    # This is what lvlm_bot will use:
    if image is not None:
        is_mllama_local = getattr(model, 'is_mllama', False) if model is not None else False
        is_qwen_vl_local = getattr(model, 'is_qwen_vl', False) if model is not None else False
        # Both Mllama and Qwen VL use smaller image resolution handling
        state.image = process_image(image, image_process_mode, return_pil=True, is_mllama=is_mllama_local or is_qwen_vl_local)
    else:
        state.image = None

    return (state, to_gradio_chatbot(state), "", None)


@spaces.GPU
def lvlm_bot(state, temperature, top_p, max_new_tokens):
    prompt = state.prompt
    image = state.image

    is_mllama = getattr(model, 'is_mllama', False)
    is_qwen_vl = getattr(model, 'is_qwen_vl', False)

    # 🔒 Only pass images if we really have a PIL image
    if isinstance(image, PILImage.Image):
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(model.device)
    else:
        # Text-only case: do NOT pass images at all
        inputs = processor(
            text=prompt,
            return_tensors="pt",
        ).to(model.device)
    
    input_ids = inputs.input_ids
    
    # Find image token index - different for each model type
    # Initialize grid dimensions (will be set for Qwen VL)
    image_grid_hw = None
    
    if is_qwen_vl:
        # Qwen VL uses image_token_id from config (typically 151655)
        # The image tokens are embedded inline in the sequence
        image_token_id = getattr(model.config, 'image_token_id', 151655)
        img_idx_matches = torch.where(input_ids == image_token_id)[1]
        if len(img_idx_matches) > 0:
            img_idx = img_idx_matches[0].item()
            # Count how many image tokens there are (for Qwen VL, this varies based on image size)
            num_image_tokens = len(img_idx_matches)
            logger.info(f"Qwen VL: Found {num_image_tokens} image tokens starting at index {img_idx}")
        else:
            # Try alternative: look for vision_start token
            vision_start_id = getattr(model.config, 'vision_start_token_id', None)
            if vision_start_id is not None:
                start_matches = torch.where(input_ids == vision_start_id)[1]
                if len(start_matches) > 0:
                    img_idx = start_matches[0].item() + 1  # +1 to skip the vision_start token itself
                else:
                    logger.warning("Could not find vision tokens in input_ids for Qwen VL model")
                    img_idx = 0
            else:
                logger.warning("Could not find image token in input_ids for Qwen VL model")
                img_idx = 0
            # Default number of image tokens - will be recalculated based on actual input
            num_image_tokens = 256  # Typical for Qwen VL with standard image size
        
        # Get actual grid dimensions from image_grid_thw (temporal, height, width)
        # NOTE: image_grid_thw gives dimensions BEFORE spatial merge (2x2 pooling)
        # The actual number of image tokens is after the merge
        if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
            grid_thw = inputs.image_grid_thw[0]  # First image
            raw_grid_h, raw_grid_w = grid_thw[1].item(), grid_thw[2].item()
            
            # Get spatial merge size from config (typically 2 for Qwen VL)
            spatial_merge_size = getattr(model.config.vision_config, 'spatial_merge_size', 2)
            
            # Calculate actual grid dimensions after spatial merge
            grid_h = raw_grid_h // spatial_merge_size
            grid_w = raw_grid_w // spatial_merge_size
            
            image_grid_hw = (grid_h, grid_w)
            num_image_tokens = grid_h * grid_w
            logger.info(f"Qwen VL: Image grid is {grid_h}x{grid_w} = {num_image_tokens} tokens (after {spatial_merge_size}x{spatial_merge_size} merge)")
    elif is_mllama:
        # Mllama uses <|image|> token (id 128256 typically)
        # The image token index in config may be different
        image_token_id = getattr(model.config, 'image_token_index', 128256)
        img_idx_matches = torch.where(input_ids == image_token_id)[1]
        if len(img_idx_matches) > 0:
            img_idx = img_idx_matches[0].item()
        else:
            # Fallback: look for the image token by searching common IDs
            logger.warning("Could not find image token in input_ids for Mllama model")
            img_idx = 0
        # For Mllama, the number of image tokens depends on image size and tiling
        # Typically it's much smaller than LLaVA's 576 patches
        num_image_tokens = 1  # Mllama uses cross-attention, so only placeholder token in input_ids
    else:
        img_idx = torch.where(input_ids == model.config.image_token_index)[1][0].item()
        num_image_tokens = 576  # LLaVA uses 576 image patches (24x24)
    
    do_sample = True if temperature > 0.001 else False
    # Clear previous attention weights to free memory
    model.enc_attn_weights = []
    model.enc_attn_weights_vit = []
    
    # Force garbage collection to reclaim memory
    import gc
    gc.collect()

    # Determine EOS token
    if is_qwen_vl:
        # Qwen VL uses <|im_end|> or <|endoftext|> as EOS
        eos_token_id = processor.tokenizer.eos_token_id
    elif is_mllama:
        # Mllama typically uses <|eot_id|> as the end of turn token
        eos_token_id = processor.tokenizer.eos_token_id
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'config') and model.language_model.config.model_type == "gemma":
        eos_token_id = processor.tokenizer('<end_of_turn>', add_special_tokens=False).input_ids[0]
    else:
        eos_token_id = processor.tokenizer.eos_token_id

    # Clear CUDA cache before generation to free fragmented memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    outputs = model.generate(
            **inputs, 
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            output_attentions=True,  # Required for interpretability - attention captured via hooks
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=eos_token_id
        )

    input_ids_list = input_ids.reshape(-1).tolist()
    # For Qwen VL, we may have multiple image tokens - zero out all of them
    if is_qwen_vl:
        image_token_id = getattr(model.config, 'image_token_id', 151655)
        for i, tid in enumerate(input_ids_list):
            if tid == image_token_id:
                input_ids_list[i] = 0
    else:
        input_ids_list[img_idx] = 0
    input_text = processor.tokenizer.decode(input_ids_list) # eg. "<s> You are a helpful ..."
    
    # Handle different BOS tokens
    if input_text.startswith("<s> "):
        input_text = '<s>' + input_text[4:] # Remove the first space after <s> to maintain correct length
    elif input_text.startswith("<|begin_of_text|> "):
        input_text = '<|begin_of_text|>' + input_text[len('<|begin_of_text|> '):]
    elif input_text.startswith("<|im_start|> "):
        # Qwen VL uses <|im_start|> token
        input_text = '<|im_start|>' + input_text[len('<|im_start|> '):]
        
    input_text_tokenized = processor.tokenizer.tokenize(input_text) # eg. ['<s>', '▁You', '▁are', '▁a', '▁helpful', ... ]
    if img_idx < len(input_text_tokenized):
        input_text_tokenized[img_idx] = "average_image"
    
    output_ids = outputs.sequences.reshape(-1)[input_ids.shape[-1]:].tolist()  

    generated_text = processor.tokenizer.decode(output_ids)
    output_ids_decoded = [processor.tokenizer.decode(oid).strip() for oid in output_ids] # eg. ['The', 'man', "'", 's', 'sh', 'irt', 'is', 'yellow', '.', '</s>']
    generated_text_tokenized = processor.tokenizer.tokenize(generated_text)

    logger.info(f"Generated response: {generated_text}")
    logger.debug(f"output_ids_decoded: {output_ids_decoded}")
    logger.debug(f"generated_text_tokenized: {generated_text_tokenized}")

    # Clean up end tokens from display
    end_tokens = ['</s>', '<|eot_id|>', '<|end_of_text|>', '<|im_end|>', '<|endoftext|>']
    cleaned_text = generated_text
    for end_token in end_tokens:
        if cleaned_text.endswith(end_token):
            cleaned_text = cleaned_text[:-len(end_token)]
    state.messages[-1][-1] = cleaned_text

    tempdir = os.getenv('TMPDIR', '/tmp/')
    tempfilename = tempfile.NamedTemporaryFile(dir=tempdir)
    tempfilename.close()

    # Save input_ids and attentions
    fn_input_ids = f'{tempfilename.name}_input_ids.pt'
    torch.save(move_to_device(input_ids, device='cpu'), fn_input_ids)
    fn_attention = f'{tempfilename.name}_attn.pt'
    torch.save(move_to_device(outputs.attentions, device='cpu'), fn_attention)
    logger.info(f"Saved attention to {fn_attention}")

    # Handle relevancy map
    # tokens_for_rel = tokens_for_rel[1:]
    word_rel_map = construct_relevancy_map(
        tokenizer=processor.tokenizer, 
        model=model,
        input_ids=inputs.input_ids,
        tokens=generated_text_tokenized, 
        outputs=outputs, 
        output_ids=output_ids,
        img_idx=img_idx
    )
    fn_relevancy = f'{tempfilename.name}_relevancy.pt'
    torch.save(move_to_device(word_rel_map, device='cpu'), fn_relevancy)
    logger.info(f"Saved relevancy map to {fn_relevancy}")
    model.enc_attn_weights = []
    model.enc_attn_weights_vit = []
    # enc_attn_weights_vit = []
    # rel_maps = []

    # Reconstruct processed image - handle different pixel_values formats
    img_std = torch.tensor(processor.image_processor.image_std).view(3,1,1)
    img_mean = torch.tensor(processor.image_processor.image_mean).view(3,1,1)
    
    pixel_values = inputs.pixel_values
    # Handle different pixel_values shapes for different models
    if is_qwen_vl:
        # Qwen VL may have pixel_values with shape (batch, num_patches, channels, height, width)
        # or (batch, channels, height, width)
        if pixel_values.dim() == 5:
            # Take first patch
            pixel_values = pixel_values[0, 0]
        elif pixel_values.dim() == 4:
            pixel_values = pixel_values[0]
        else:
            pixel_values = pixel_values[0]
    elif is_mllama and pixel_values.dim() > 4:
        # Mllama may have different pixel_values shape (e.g., with aspect ratio bins)
        # Take the first tile/aspect ratio
        pixel_values = pixel_values[0, 0]
    elif pixel_values.dim() == 4:
        pixel_values = pixel_values[0]
    else:
        pixel_values = pixel_values[0]
    
    img_recover = pixel_values.cpu() * img_std + img_mean
    img_recover = to_pil_image(img_recover)

    state.recovered_image = img_recover
    state.input_text_tokenized = input_text_tokenized
    state.output_ids_decoded = output_ids_decoded 
    state.attention_key = tempfilename.name
    state.image_idx = img_idx
    state.is_mllama = is_mllama
    state.is_qwen_vl = is_qwen_vl
    state.num_image_tokens = num_image_tokens
    # For Qwen VL: store actual grid dimensions (height, width) for proper aspect ratio
    # This is None for LLaVA (which uses fixed 24x24) and Mllama
    state.image_grid_hw = image_grid_hw

    return state, to_gradio_chatbot(state) 


def build_demo(args, embed_mode=False):
    global model
    global processor
    global system_prompt
    global ROLE0
    global ROLE1
    global N_LAYERS

    if model is None:
        processor, model = get_processor_model(args)

    # Set model-specific configurations
    is_mllama = getattr(model, 'is_mllama', False)
    is_qwen_vl = getattr(model, 'is_qwen_vl', False)
    
    if is_qwen_vl:
        # Qwen VL models: language model is accessed through model.model.language_model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            N_LAYERS = len(model.model.language_model.layers)
        elif hasattr(model, 'language_model'):
            N_LAYERS = len(model.language_model.layers)
        else:
            N_LAYERS = 28  # Default for Qwen3-VL-2B
        system_prompt = ''  # Qwen VL uses chat template
        ROLE0 = 'user'
        ROLE1 = 'assistant'
        logger.info(f"Qwen VL model detected with {N_LAYERS} layers")
    elif is_mllama:
        # Llama 3.2 Vision has 40 layers for 11B model
        N_LAYERS = len(model.language_model.layers)
        system_prompt = ''  # Mllama uses chat template
        ROLE0 = 'user'
        ROLE1 = 'assistant'
    elif 'gemma' in args.model_name_or_path:
        system_prompt = ''
        ROLE0 = 'user'
        ROLE1 = 'model'
        # Get number of layers from model
        language_model_layers = getattr(model.language_model, 'model', model.language_model).layers
        N_LAYERS = len(language_model_layers)
    else:
        # Get number of layers from model
        language_model_layers = getattr(model.language_model, 'model', model.language_model).layers
        N_LAYERS = len(language_model_layers)

    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LVLM-Interpret") as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column(scale=6):
                    
                    imagebox = gr.ImageEditor(type="pil", height=400, elem_id="image_canvas")
                    

                    with gr.Accordion("Parameters", open=False) as parameter_row:
                        image_process_mode = gr.Radio(
                            ["Crop", "Resize", "Pad", "Default"],
                            value="Default",
                            label="Preprocess for non-square image", visible=True
                        )
                        temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                        max_output_tokens = gr.Slider(minimum=0, maximum=512, value=32, step=32, interactive=True, label="Max new output tokens",)


                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(
                                    elem_id="chatbot",
                                    label="Chatbot",
                                    height=400,
                                    sanitize_html=False,
                                )
                    with gr.Row():
                        with gr.Column(scale=8):
                            textbox.render()
                        with gr.Column(scale=1, min_width=50):
                            submit_btn = gr.Button(value="Send", variant="primary")
                    with gr.Row(elem_id="buttons") as button_row:
                        clear_btn = gr.Button(value="🗑️  Clear", interactive=True, visible=True)

            # with gr.Row():
            #     with gr.Column(scale=6):
                    
            #         gr.Examples(examples=[
            #             [f"{CUR_DIR}/examples/extreme_ironing.jpg", "What color is the man's shirt?"],
            #             [f"{CUR_DIR}/examples/waterview.jpg", "What is in the top left of this image?"],
            #             [f"{CUR_DIR}/examples/MMVP_34.jpg", "Is the butterfly's abdomen visible in the image?"],
            #         ], inputs=[imagebox, textbox])

            #     with gr.Column(scale=6):
            #         gr.Examples(examples=[
            #             [f"{CUR_DIR}/examples/MMVP_84.jpg", "Is the door of the truck cab open?"],
            #             [f"{CUR_DIR}/examples/MMVP_173.jpg", "Is the decoration on the Easter egg flat or raised?"],
            #             [f"{CUR_DIR}/examples/MMVP_279.jpg", "Is the elderly person standing or sitting in the picture?"],
            #         ], inputs=[imagebox, textbox])

        with gr.Tab("Attention analysis"):
            with gr.Row():
                with gr.Column(scale=3):
                    # attn_ana_layer = gr.Slider(1, 100, step=1, label="Layer")
                    attn_modality_select = gr.Dropdown(
                            choices=['Image-to-Answer', 'Question-to-Answer'],
                            value='Image-to-Answer',
                            interactive=True,
                            show_label=False,
                            container=False
                        )
                    attn_ana_submit = gr.Button(value="Plot attention matrix", interactive=True)
                with gr.Column(scale=6):
                    attn_ana_plot = gr.Plot(label="Attention plot")

        attn_ana_submit.click(
                plot_attention_analysis,
                [state, attn_modality_select],
                [state, attn_ana_plot]
            )

        with gr.Tab("Attentions"):
            with gr.Row():
                attn_select_layer = gr.Slider(1, N_LAYERS, value=32, step=1, label="Layer")
            with gr.Row():
                with gr.Column(scale=3):
                    imagebox_recover = gr.Image(type="pil", label='Preprocessed image', interactive=False)

                    generated_text = gr.HighlightedText(
                        label="Generated text (tokenized)",
                        combine_adjacent=False,
                        interactive=True,
                        color_map={"label": "green"}
                    )
                    with gr.Row():
                        attn_reset = gr.Button(value="Reset tokens", interactive=True)
                        attn_submit = gr.Button(value="Plot attention", interactive=True)

                with gr.Column(scale=9):
                    i2t_attn_head_mean_plot = gr.Plot(label="Image-to-Text attention average per head")
                    i2t_attn_gallery = gr.Gallery(type="pil", label='Attention heatmaps', columns=8, interactive=False)

            box_states = gr.Dataframe(type="numpy", datatype="bool", row_count=24, col_count=24, visible=False) 
            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    imagebox_recover_boxable = gr.Image(label='Patch Selector')
                    attn_ana_head= gr.Slider(1, 40, step=1, label="Head Index")
            
                    reset_boxes_btn = gr.Button(value="Reset patch selector")
                    attn_ana_submit_2 = gr.Button(value="Plot attention matrix", interactive=True)
                
                with gr.Column(scale=9):
                    t2i_attn_head_mean_plot = gr.Plot(label="Text-to-Image attention average per head")
                    attn_ana_plot_2 = gr.Plot(scale=2, label="Attention plot",container=True)

        reset_boxes_btn.click(
            handle_box_reset, 
            [imagebox_recover,box_states], 
            [imagebox_recover_boxable, box_states]
        )
        imagebox_recover_boxable.select(boxes_click_handler, [imagebox_recover,box_states], [imagebox_recover_boxable, box_states])
        
        attn_reset.click(
            reset_tokens,
            [state],
            [generated_text]
        )

        attn_ana_submit_2.click(
            plot_text_to_image_analysis,
            [state, attn_select_layer, box_states, attn_ana_head ],
            [state, attn_ana_plot_2, t2i_attn_head_mean_plot]
        )
        

        attn_submit.click(
            handle_attentions_i2t,
            [state, generated_text, attn_select_layer],
            [generated_text, imagebox_recover, i2t_attn_gallery, i2t_attn_head_mean_plot]
        )

        with gr.Tab("Relevancy"):
            with gr.Row():
                relevancy_token_dropdown = gr.Dropdown(
                    choices=['llama','vit','all'],
                    value='llama',
                    interactive=True,
                    show_label=False,
                    container=False
                )
                relevancy_submit = gr.Button(value="Plot relevancy", interactive=True)
            with gr.Row():
                relevancy_gallery = gr.Gallery(type="pil", label='Input image relevancy heatmaps', columns=8, interactive=False)
            with gr.Row():
                relevancy_txt_gallery = gr.Gallery(type="pil", label='Image-text relevancy comparison', columns=8, interactive=False)
                #gr.Plot(label='Input text Relevancy heatmaps') 
            with gr.Row():
                relevancy_highlightedtext = gr.HighlightedText(
                        label='Tokens with high relevancy to image'
                    )

        relevancy_submit.click(
            lambda state, relevancy_token_dropdown: handle_relevancy(state, relevancy_token_dropdown, incude_text_relevancy=True),
            #handle_relevancy,
            [state, relevancy_token_dropdown],
            [relevancy_gallery],
        )
        relevancy_submit.click(
            handle_text_relevancy,
            [state, relevancy_token_dropdown],
            [relevancy_txt_gallery, relevancy_highlightedtext]
        )

        enable_causality = False
        with gr.Tab("Causality"):
            gr.Markdown(
                """
                ### *Coming soon*
                """
            )
            state_causal_explainers = gr.State()
            with gr.Row(visible=enable_causality):
                causality_dropdown = gr.Dropdown(
                    choices=[],
                    interactive=True,
                    show_label=False,
                    container=False,
                    scale=2,
                )
                causality_submit = gr.Button(value="Learn Causal Structures", interactive=True, variant='primary', scale=1)
            with gr.Row(visible=enable_causality):
                with gr.Accordion("Hyper Parameters", open=False) as causal_parameters_row:
                        with gr.Row():
                            with gr.Column(scale=2):
                                # search_rad_slider= gr.Slider(1, 5, step=1, value=3, label="Search Radius", 
                                #                              info="The maximal distance on the graph from the explained token.",)
                                att_th_slider = gr.Slider(minimum=0.0001, maximum=1-0.0001, value=0.005, step=0.0001, interactive=True, label="Raw Attention Threshold",
                                                          info="A threshold for selecting tokens to be graph nodes.",)
                            with gr.Column(scale=2):
                                alpha_slider = gr.Slider(minimum=1e-7, maximum=1e-2, value=1e-5, step=1e-7, interactive=True, label="Statistical Test Threshold (alpha)",
                                                         info="A threshold for the statistical test of conditional independence.",)
                                # dof_slider = gr.Slider(minimum=32, maximum=1024, value=128, step=1, interactive=True, label="Degrees of Freedom",
                                #                        info="Degrees of freedom of correlation matrix.")
            with gr.Row(visible=enable_causality):
                pds_plot = gr.Image(type="pil", label='Preprocessed image')
                causal_head_gallery = gr.Gallery(type="pil", label='Causal Head Graph', columns=8, interactive=False)
            with gr.Row(visible=enable_causality):
                causal_head_slider = gr.Slider(minimum=0, maximum=31, value=1, step=1, interactive=True, label="Head Selection")
                causal_head_submit = gr.Button(value="Plot Causal Head", interactive=True, scale=1)
            with gr.Row(visible=enable_causality):
                causality_gallery = gr.Gallery(type="pil", label='Causal Heatmaps', columns=8, interactive=False)
    
        causal_head_submit.click(
            handle_causal_head,
            [state, state_causal_explainers, causal_head_slider, causality_dropdown],
            [causal_head_gallery, pds_plot]
        )
        
        causality_submit.click(
            handle_causality,
            [state, state_causal_explainers, causality_dropdown, alpha_slider, att_th_slider],
            [causality_gallery, state_causal_explainers]
        )

        if not embed_mode:
            gr.Markdown(tos_markdown)

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox, imagebox_recover, generated_text, i2t_attn_gallery ] ,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            lvlm_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot] ,
        ).then(
            attn_update_slider,
            [state],
            [state, attn_select_layer]
        ).then(
            causality_update_dropdown,
            [state],
            [state, causality_dropdown]
        )
        # .then(
        #     handle_box_reset, 
        #     [imagebox_recover,box_states], 
        #     [imagebox_recover_boxable, box_states]
        # ).then(
        #     handle_attentions_i2t,
        #     [state, generated_text, attn_select_layer],
        #     [generated_text, imagebox_recover, i2t_attn_gallery, i2t_attn_head_mean_plot]
        # ).then(
        #     clear_canvas,
        #     [],
        #     [imagebox]
        # ).then(
        #     handle_relevancy,
        #     [state, relevancy_token_dropdown],
        #     [relevancy_gallery]
        # ).then(
        #     handle_text_relevancy,
        #     [state, relevancy_token_dropdown],
        #     [relevancy_txt_gallery, relevancy_highlightedtext]
        # )
        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            lvlm_bot,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot],
        ).then(
            attn_update_slider,
            [state],
            [state, attn_select_layer]
        ).then(
            causality_update_dropdown,
            [state],
            [state, causality_dropdown]
        )
        # .then(
        #     causality_update_dropdown,
        #     [state],
        #     [causality_dropdown]
        # ).then(
        #     handle_box_reset, 
        #     [imagebox_recover,box_states], 
        #     [imagebox_recover_boxable, box_states]
        # ).then(
        #      plot_attention_analysis,
        #      [state, attn_modality_select],
        #      [state, attn_ana_plot]
        # ).then(
        #     handle_relevancy,
        #     [state, relevancy_token_dropdown],
        #     [relevancy_gallery]
        # ).then(
        #     handle_text_relevancy,
        #     [state, relevancy_token_dropdown],
        #     [relevancy_txt_gallery, relevancy_highlightedtext]
        # )
        

    return demo