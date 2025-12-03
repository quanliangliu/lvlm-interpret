import os, sys
llava_home = os.getenv('LLAVA_HOME')
if llava_home:
    sys.path.append(llava_home)

from collections import defaultdict
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
import gradio as gr
import PIL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba

import seaborn
from PIL import Image, ImageDraw
import pandas as pd
from scipy import stats

import logging

logger = logging.getLogger(__name__)
cmap = plt.get_cmap('jet')
separators_list = ['.',',','?','!', ':', ';', '</s>', '/', '!', '(', ')', '[', ']', '{', '}', '<', '>', '|', '\\', '-', '_', '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ' ', '\t', '\n', '\r', '\x0b', '\x0c']

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

def convert_token2word(R_i_i, tokens, separators_list):
    current_count = 1
    current_rel_map = 0
    word_rel_maps = {}
    current_word = ""
    for token, rel in zip(tokens, R_i_i):
        if not token.startswith('▁') and token not in separators_list:
            current_word += token
            current_rel_map += rel
            current_count += 1
        else:
            # Otherwise, store the current word's relevancy map and start a new word
            word_rel_maps[current_word] = current_rel_map / current_count
            current_word = token
            current_rel_map = rel
            current_count = 1
    return list(word_rel_maps.keys()), torch.Tensor(list(word_rel_maps.values()))

def draw_heatmap_on_image(mat, img_recover, normalize=True):
    if normalize:
        mat_min = mat.min()
        mat_max = mat.max()
        mat_range = mat_max - mat_min
        if mat_range > 1e-8:
            mat = (mat - mat_min) / mat_range
        else:
            # All values are the same, create a uniform mid-value
            mat = np.ones_like(mat) * 0.5
            logger.warning(f"Heatmap has no variation (min={mat_min}, max={mat_max}), using uniform values")
    
    # Ensure mat is a valid numpy array
    if hasattr(mat, 'cpu'):
        mat = mat.cpu().numpy()
    mat = np.array(mat, dtype=np.float32)
    
    # Log heatmap statistics for debugging
    logger.debug(f"Heatmap stats: min={mat.min():.4f}, max={mat.max():.4f}, mean={mat.mean():.4f}, shape={mat.shape}")
    
    mat = cmap(mat)
    mat = Image.fromarray((mat[:, :, :3] * 255).astype(np.uint8)).resize((336,336), Image.BICUBIC)
    mat.putalpha(180)  # Increased alpha for more visible overlay
    img_overlay_attn = img_recover.copy().convert('RGBA')
    mat = mat.convert('RGBA')
    img_overlay_attn = Image.alpha_composite(img_overlay_attn, mat)
    
    return img_overlay_attn.convert('RGB')

def attn_update_slider(state):
    fn_attention = state.attention_key + '_attn.pt'
    attentions = torch.load(fn_attention, mmap=True)
    num_layers = len(attentions[0])
    return state, gr.Slider(1, num_layers, value=num_layers, step=1, label="Layer")


def handle_attentions_i2t(state, highlighted_text, layer_idx=32, token_idx=0):
    '''
        Draw attention heatmaps and return as a list of PIL images
    '''
    logger.info(f"handle_attentions_i2t called with layer_idx={layer_idx}")

    if not hasattr(state, 'attention_key'):
        logger.warning("No attention_key found in state")
        return None, None, [], None
    
    recovered_image = getattr(state, 'recovered_image', None)
    if recovered_image is None:
        logger.warning("No recovered_image found - cannot plot attention heatmaps")
        return None, None, [], None
        
    layer_idx -= 1 
    fn_attention = state.attention_key + '_attn.pt'
    img_idx = state.image_idx
    
    # Get model-specific parameters
    is_mllama = getattr(state, 'is_mllama', False)
    num_image_tokens = getattr(state, 'num_image_tokens', 576)
    
    # Calculate grid dimensions for visualization
    if num_image_tokens == 576:
        grid_h, grid_w = 24, 24
    else:
        # For Mllama or other models - try to infer square grid
        grid_h = grid_w = int(np.sqrt(num_image_tokens))
        if grid_h * grid_w != num_image_tokens:
            logger.warning(f"Non-square image token count: {num_image_tokens}, using grid_size={grid_h}")

    if highlighted_text is not None:
        decoded_tokens = getattr(state, 'output_ids_decoded', [])
        token_idx_list = []
        decoded_pointer = 0
        generated_text = []

        for item in highlighted_text:
            if isinstance(item, dict):
                token = item.get('token', '')
                label = item.get('class_or_confidence')
            elif isinstance(item, (list, tuple)) and item:
                token = item[0]
                label = item[1] if len(item) > 1 else None
            else:
                token, label = '', None
            generated_text.append((token, label))

            if not token or token.isspace():
                continue

            if decoded_pointer >= len(decoded_tokens):
                logger.warning(
                    f"Received more tokens ({decoded_pointer + 1}) from UI than available decoded outputs ({len(decoded_tokens)})."
                )
                continue

            if label is not None:
                token_idx_list.append(decoded_pointer)

            decoded_pointer += 1

        if not token_idx_list:
            logger.info("No tokens selected – defaulting to first generated token.")
            token_idx_list = [0] if decoded_tokens else []

        if not token_idx_list:
            gr.Error("No generated tokens available for attention.")
            return None, None, [], None
    else:
        if not getattr(state, 'output_ids_decoded', []):
            gr.Error("No generated tokens available for attention.")
            return None, None, [], None

        token_idx_list = [0]

        generated_text = []
        for text in state.output_ids_decoded:
            generated_text.extend([(text, None), (' ', None)])
        

    if not os.path.exists(fn_attention):
        logger.error(f'Attention file not found: {fn_attention}')
        gr.Error('Attention file not found. Please re-run query.')
        return None, None, [], None
    
    logger.info(f'Loading attention from {fn_attention}...')
    attentions = torch.load(fn_attention)
    logger.info(f'Loaded attention with {len(attentions)} tokens')
    
    if len(attentions) == len(state.output_ids_decoded):
        gr.Error('Mismatch between lengths of attentions and output tokens')
    batch_size, num_heads, inp_seq_len, seq_len = attentions[0][0].shape
    logger.info(f'Processing {num_heads} heads, {len(token_idx_list)} selected tokens')
    cmap = plt.get_cmap('jet')

    img_attn_list = []
    img_attn_mean = []
    for head_idx in range(num_heads):
        img_attn = None
        for token_idx in token_idx_list:
            if token_idx >= len(attentions):
                logger.info(f'token index {token_idx} out of bounds')
                continue
            mh_attention = attentions[token_idx][layer_idx]
            batch_size, num_heads, inp_seq_len, seq_len = mh_attention.shape
            if inp_seq_len > 1:
                mh_attention = mh_attention[:,:,-1,:]
            mh_attention = mh_attention.squeeze()
            
            # Handle different image token counts
            img_end_idx = min(img_idx + num_image_tokens, seq_len)
            actual_img_tokens = img_end_idx - img_idx
            
            if actual_img_tokens > 0:
                img_attn_raw = mh_attention[head_idx, img_idx:img_end_idx].float().cpu().numpy()
                # Reshape to grid using actual dimensions
                from scipy.ndimage import zoom
                target_size = 24  # Standard output size for visualization
                
                if actual_img_tokens == num_image_tokens and actual_img_tokens == grid_h * grid_w:
                    # Reshape to actual grid dimensions
                    img_attn_grid = img_attn_raw.reshape(grid_h, grid_w)
                    # Resize to target visualization size
                    zoom_factors = (target_size / grid_h, target_size / grid_w)
                    img_attn_token = zoom(img_attn_grid, zoom_factors, order=1)
                else:
                    # Fallback: interpolate to standard visualization size
                    img_attn_token = np.zeros((target_size, target_size))
                    if actual_img_tokens > 0:
                        # Try to infer grid dimensions
                        inferred_h = inferred_w = int(np.ceil(np.sqrt(actual_img_tokens)))
                        # Pad if needed
                        padded = np.zeros(inferred_h * inferred_w)
                        padded[:actual_img_tokens] = img_attn_raw
                        padded = padded.reshape(inferred_h, inferred_w)
                        zoom_factors = (target_size / inferred_h, target_size / inferred_w)
                        img_attn_token = zoom(padded, zoom_factors, order=1)
            else:
                img_attn_token = np.zeros((24, 24))

            if img_attn is None:
                img_attn = img_attn_token
            else:
                img_attn += img_attn_token
        img_attn /= len(token_idx_list)
        img_overlay_attn = draw_heatmap_on_image(img_attn, recovered_image)

        img_attn_list.append((img_overlay_attn, f'Head_{head_idx}'))

        # Calculate mean attention per head
        img_attn /= (img_attn.max() + 1e-8)
        img_attn_mean.append(img_attn.mean())
    
    img_attn_list = [x for _, x in sorted(zip(img_attn_mean, img_attn_list), key=lambda pair: pair[0], reverse=True)]

    fig = plt.figure(figsize=(10, 3))
    ax = seaborn.heatmap([img_attn_mean], 
        linewidths=.3, square=True, cbar_kws={"orientation": "horizontal", "shrink":0.3}
    )
    ax.set_xlabel('Head number')
    ax.set_title(f"Mean Attention between the image and the token {[state.output_ids_decoded[tok] for tok in token_idx_list]} for layer {layer_idx+1}")

    fig.tight_layout()

    return generated_text, recovered_image, img_attn_list, fig

def handle_relevancy(state, type_selector,incude_text_relevancy=False):
    incude_text_relevancy = True
    logger.debug(f'incude_text_relevancy: {incude_text_relevancy}')

    if not hasattr(state, 'attention_key'):
        return []
    
    fn_attention = state.attention_key + '_relevancy.pt'
    recovered_image = state.recovered_image
    img_idx = state.image_idx
    
    # Get model-specific parameters
    is_mllama = getattr(state, 'is_mllama', False)
    num_image_tokens = getattr(state, 'num_image_tokens', 576)
    
    # Calculate grid dimensions
    if num_image_tokens == 576:
        grid_h, grid_w = 24, 24
    else:
        grid_h = grid_w = int(np.sqrt(num_image_tokens)) if num_image_tokens > 1 else 24

    word_rel_maps = torch.load(fn_attention)
    if type_selector not in word_rel_maps:
        logger.warning(f'{type_selector} not in keys: {word_rel_maps.keys()}')
        return []

    word_rel_map = word_rel_maps[type_selector]
    image_list = []
    i = 0
    logger.info(f"Processing {len(word_rel_map)} relevancy maps for type={type_selector}, img_idx={img_idx}, num_image_tokens={num_image_tokens}")
    for rel_key, rel_map in word_rel_map.items():
        i+=1
        if rel_key in separators_list:
            continue
        logger.debug(f"Processing rel_key={rel_key}, rel_map.shape={rel_map.shape}")
        condition_check = (rel_map.shape[-1] != num_image_tokens + 1) and img_idx
        logger.debug(f"Condition: rel_map.shape[-1]={rel_map.shape[-1]}, num_image_tokens+1={num_image_tokens+1}, img_idx={img_idx}, result={condition_check}")
        if condition_check:
            if not incude_text_relevancy:
                img_end_idx = min(img_idx + num_image_tokens, rel_map.shape[-1])
                actual_img_tokens = img_end_idx - img_idx
                rel_map_img = rel_map[-1,:][img_idx:img_end_idx].float().cpu().numpy()
                logger.debug(f"rel_map_img stats: min={rel_map_img.min():.4f}, max={rel_map_img.max():.4f}, shape={rel_map_img.shape}")
                
                # Reshape to grid using actual dimensions, handling non-square grids
                from scipy.ndimage import zoom
                target_size = 24  # Standard output size for visualization
                
                if actual_img_tokens == num_image_tokens and actual_img_tokens == grid_h * grid_w:
                    # Reshape to actual grid dimensions
                    rel_map_grid = rel_map_img.reshape(grid_h, grid_w)
                    # Resize to target visualization size
                    zoom_factors = (target_size / grid_h, target_size / grid_w)
                    rel_map = zoom(rel_map_grid, zoom_factors, order=1)
                else:
                    # Fallback: interpolate to standard visualization size
                    if actual_img_tokens > 0:
                        inferred_h = inferred_w = int(np.ceil(np.sqrt(actual_img_tokens)))
                        padded = np.zeros(inferred_h * inferred_w)
                        padded[:actual_img_tokens] = rel_map_img
                        padded = padded.reshape(inferred_h, inferred_w)
                        zoom_factors = (target_size / inferred_h, target_size / inferred_w)
                        rel_map = zoom(padded, zoom_factors, order=1)
                    else:
                        rel_map = np.zeros((target_size, target_size))
                normalize_image_tokens = True
            if incude_text_relevancy:
                input_text_tokenized = state.input_text_tokenized
                input_text_tokenized_len = int(len(input_text_tokenized))
                img_idx_int = int(img_idx)
                
                img_end_idx = min(img_idx_int + num_image_tokens, rel_map.shape[-1])
                text_start_idx = img_end_idx + 3
                text_end_idx = min(num_image_tokens + input_text_tokenized_len - 1 - 5, rel_map.shape[-1])
                
                logger.debug(f"incude_text_relevancy branch: img_idx={img_idx_int}, img_end_idx={img_end_idx}, text_start_idx={text_start_idx}, text_end_idx={text_end_idx}")
                
                rel_maps_img = rel_map[-1,:][img_idx_int:img_end_idx]
                rel_maps_text = rel_map[-1,:][text_start_idx:text_end_idx] if text_start_idx < text_end_idx else torch.tensor([])
                rel_maps_no_system = torch.cat((rel_maps_img, rel_maps_text))
                
                logger.debug(f'shape of rel_maps_no_system: {rel_maps_no_system.shape}, raw stats: min={rel_maps_no_system.min():.4f}, max={rel_maps_no_system.max():.4f}')
                rel_maps_no_system = (rel_maps_no_system - rel_maps_no_system.min()) / (rel_maps_no_system.max() - rel_maps_no_system.min() + 1e-8)
                
                # Get image portion and reshape using actual grid dimensions
                actual_img_tokens = img_end_idx - img_idx_int
                rel_map_img = rel_maps_no_system[:actual_img_tokens].float().cpu().numpy()
                logger.debug(f"actual_img_tokens={actual_img_tokens}, rel_map_img stats: min={rel_map_img.min():.4f}, max={rel_map_img.max():.4f}")
                
                from scipy.ndimage import zoom
                target_size = 24  # Standard output size for visualization
                
                if actual_img_tokens == num_image_tokens and actual_img_tokens == grid_h * grid_w:
                    # Reshape to actual grid dimensions
                    rel_map_grid = rel_map_img.reshape(grid_h, grid_w)
                    # Resize to target visualization size
                    zoom_factors = (target_size / grid_h, target_size / grid_w)
                    rel_map = zoom(rel_map_grid, zoom_factors, order=1)
                else:
                    # Fallback: interpolate to standard visualization size
                    if actual_img_tokens > 0:
                        inferred_h = inferred_w = int(np.ceil(np.sqrt(actual_img_tokens)))
                        padded = np.zeros(inferred_h * inferred_w)
                        padded[:actual_img_tokens] = rel_map_img
                        padded = padded.reshape(inferred_h, inferred_w)
                        zoom_factors = (target_size / inferred_h, target_size / inferred_w)
                        rel_map = zoom(padded, zoom_factors, order=1)
                    else:
                        rel_map = np.zeros((target_size, target_size))
                # Always normalize the image portion since it may not span 0-1 after extraction
                normalize_image_tokens = True
        else:
            # ViT relevancy map
            rel_map = rel_map[0,1:].float().cpu().numpy()
            num_vit_tokens = len(rel_map)
            vit_grid = int(np.sqrt(num_vit_tokens))
            if vit_grid * vit_grid == num_vit_tokens:
                rel_map = rel_map.reshape(vit_grid, vit_grid)
            else:
                from scipy.ndimage import zoom
                side = int(np.ceil(np.sqrt(num_vit_tokens)))
                padded = np.zeros(side * side)
                padded[:num_vit_tokens] = rel_map
                padded = padded.reshape(side, side)
                zoom_factor = 24 / side
                rel_map = zoom(padded, zoom_factor, order=1)
            normalize_image_tokens = True
        logger.info(f"Before draw_heatmap: rel_map.shape={rel_map.shape}, rel_map stats: min={rel_map.min():.4f}, max={rel_map.max():.4f}, normalize={normalize_image_tokens}")
        rel_map = draw_heatmap_on_image(rel_map, recovered_image, normalize=normalize_image_tokens)
        # strip _ from all rel keys
        rel_key = rel_key.strip('▁').strip('_')
        image_list.append( (rel_map, rel_key))

    return image_list

def grid_size(len):
    n_columns = 3 if len < 16 else 4
    n_rows = int(np.ceil(len / n_columns))
    return (n_rows, n_columns)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def handle_text_relevancy(state, type_selector):
    if type_selector != "llama":
        return [], []
    else:
        tokens = state.output_ids_decoded
        fn_attention = state.attention_key + '_relevancy.pt'
        img_idx = state.image_idx
        input_text_tokenized = state.input_text_tokenized
        word_rel_maps = torch.load(fn_attention)
        
        # Get model-specific parameters
        num_image_tokens = getattr(state, 'num_image_tokens', 576)
        
        input_text_tokenized_all = input_text_tokenized.copy()
        # loop over all output tokens
        word_rel_map = word_rel_maps["llama_token"]
        # grid_size_temp = grid_size(len(rel_scores))
        all_figs = []
        highlighted_tokens = []
        for word, rel_map in word_rel_map.items():
            if word in separators_list:
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            # if the token is not a separator
            # if i < len(tokens) and tokens[i] not in separators_list:
            img_end_idx = min(img_idx + num_image_tokens, rel_map.shape[-1])
            img_avg_rel = rel_map[-1,:][img_idx:img_end_idx].mean()
            img_max_rel = rel_map[-1,:][img_idx:img_end_idx].max()
            logger.debug(f'img_avg_rel for token {word}: {img_avg_rel}')
            # exclude the image tokens from the rel_scores[i] and replace all of them by a single value of the average relevancy for the image
            current_relevency = rel_map[-1,:][:img_idx].clone()
            # add the average relevancy for the image to the current_relevency tensor
            current_relevency = torch.cat((current_relevency, img_avg_rel.unsqueeze(0)))
            text_end_idx = min(img_end_idx + len(input_text_tokenized_all) - img_idx - 1, rel_map.shape[-1])
            current_relevency = torch.cat((current_relevency, rel_map[-1,:][img_end_idx:text_end_idx]))
            current_relevency = current_relevency.cpu()
            logger.debug(f'shape of text relevancy map: {rel_map[-1,:].shape}')
            #rel_score_text = rel_scores[i][-1,:][:img_idx]
            
            # Adjust assertion for different model types
            expected_len = len(input_text_tokenized_all) - num_image_tokens + 1  # +1 for the average image token
            if len(current_relevency) != expected_len:
                logger.warning(f"Length mismatch: relevancy={len(current_relevency)}, expected={expected_len}")
                # Truncate or pad as needed
                if len(current_relevency) > expected_len:
                    current_relevency = current_relevency[:expected_len]
            
            # After collapsing image tokens to single avg value, structure is:
            # [tokens_before_img | avg_img_token | tokens_after_img]
            # We want to skip system prompt (before img_idx) and keep user text (after collapsed img token)
            # The collapsed image token is at index img_idx, so user text starts at img_idx + 1
            
            # Skip system prompt tokens (before image) but keep image avg and everything after
            # Also skip last few tokens (typically EOS/padding)
            offset = img_idx + 1  # Start after the collapsed image token
            end_offset = max(offset, len(current_relevency) - 2)  # Keep more tokens, only skip last 2
            
            if offset < len(current_relevency) and offset < end_offset:
                current_relevency = current_relevency[offset:end_offset]
                # Align input_text_tokenized with the same offset
                # Note: input_text_tokenized_all still has original structure with all image tokens
                # We need to map: offset in collapsed -> offset in original
                # collapsed[img_idx+1] corresponds to original[img_idx + num_image_tokens]
                original_offset = img_idx + num_image_tokens
                original_end = original_offset + (end_offset - offset)
                input_text_tokenized = input_text_tokenized_all[original_offset:original_end] if original_offset < len(input_text_tokenized_all) else []
                logger.debug(f"Relevancy slice: offset={offset}, end_offset={end_offset}, original_offset={original_offset}, len(input_text_tokenized)={len(input_text_tokenized)}")
            else:
                # Fallback: use all tokens
                current_relevency = current_relevency
                input_text_tokenized = input_text_tokenized_all
                logger.debug("Using all tokens for relevancy (fallback)")
                
            if len(input_text_tokenized) > 0 and len(current_relevency) > 0:
                input_text_tokenized_word, current_relevency_word = convert_token2word(current_relevency, input_text_tokenized, separators_list)
            else:
                input_text_tokenized_word, current_relevency_word = [], torch.tensor([])

            if len(current_relevency_word) > 0:
                current_relevency_word_topk = current_relevency_word.topk(min(3, len(current_relevency_word)))
                max_rel_scores = current_relevency_word_topk.values
                max_rel_scores = torch.cat((max_rel_scores, img_max_rel.unsqueeze(0).cpu()))
                max_rel_scores_idx = current_relevency_word_topk.indices
                max_input_token = [input_text_tokenized_word[j].lstrip('▁').lstrip('_') for j in max_rel_scores_idx if j < len(input_text_tokenized_word)]
            else:
                max_rel_scores = img_max_rel.unsqueeze(0).cpu()
                max_input_token = []

            # Image to text relevancy ratio
            img_text_rel_value = stats.percentileofscore(max_rel_scores, img_max_rel.item(), kind='strict') / 100 if len(max_rel_scores) > 0 else 0.5

            highlighted_tokens.extend(
                [
                    (word.strip('▁'), float(img_text_rel_value)),
                    (" ", None)
                ]
            )

            max_input_token.append("max_img")
            ax.bar(max_input_token, max_rel_scores)
            # ax.set_xticklabels(max_input_token, fontsize=12)

            # save the plot per each output token
            # make part of the title bold
            ax.set_title(f'Output Token: {word.strip("▁").strip("_")}', fontsize=15)
            # add labels for the x and y axis
            ax.set_xlabel('Input Tokens', fontsize=15)
            ax.set_ylabel('Relevancy Score', fontsize=15)

            fig.tight_layout()

            fig_pil = fig2img(fig)
            all_figs.append(fig_pil)

        return all_figs, highlighted_tokens

def handle_image_click(image, box_grid, x, y):
    # If box_grid is empty or wrong size, create a new one
    if box_grid is None or box_grid.size == 0 or box_grid.shape != (24, 24):
        box_grid = np.zeros((24, 24), dtype=bool)
    
    # Calculate which box was clicked
    # image.size returns (width, height) for PIL images
    box_width = image.size[0] // 24   # width / 24
    box_height = image.size[1] // 24  # height / 24

    # x is horizontal (column), y is vertical (row)
    box_col = x // box_width
    box_row = y // box_height
    
    # Clamp to valid range
    box_col = min(max(box_col, 0), 23)
    box_row = min(max(box_row, 0), 23)

    # box_grid is indexed as [row][col] for consistency with numpy/matplotlib
    box_grid[box_row][box_col] = not box_grid[box_row][box_col]
    logger.info(f"Toggled patch at row={box_row}, col={box_col}, value={box_grid[box_row][box_col]}")
    
    # Add a transparent teal box to the image at the clicked location
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    indices = np.where(box_grid)
    for row, col in zip(*indices):
        draw.rectangle([(col * box_width, row * box_height), ((col + 1) * box_width, (row + 1) * box_height)], fill=(255, 100, 100, 128))

    image = Image.blend(image, overlay, alpha=0.8)

    # Return the updated image
    return image, box_grid

def handle_box_reset(input_image, box_grid): 
    # If box_grid is empty or wrong size, create a new one
    if box_grid is None or box_grid.size == 0 or box_grid.shape != (24, 24):
        box_grid = np.zeros((24, 24), dtype=bool)
    else:
        for i in range(24):
            for j in range(24):
                box_grid[i][j] = False
    try:
        to_return = input_image.copy()
    except:
        to_return = None
    return to_return, box_grid


def boxes_click_handler(image, box_grid, event: gr.SelectData):
    if event is None:
        return image, box_grid
    
    x, y = event.index[0], event.index[1]
    if x is None or y is None:
        return image, box_grid
    
    image, box_grid = handle_image_click(image, box_grid, x, y)
    return image, box_grid

def plot_attention_analysis(state, attn_modality_select):
    logger.info(f"plot_attention_analysis called with modality={attn_modality_select}")
    
    if not hasattr(state, 'attention_key'):
        logger.warning("No attention_key found in state")
        return state, None
    
    fn_attention = state.attention_key + '_attn.pt'
    recovered_image = getattr(state, 'recovered_image', None)
    img_idx = getattr(state, 'image_idx', 0)
    
    # Get model-specific parameters
    num_image_tokens = getattr(state, 'num_image_tokens', 576)
    
    # Calculate grid dimensions
    if num_image_tokens == 576:
        grid_h, grid_w = 24, 24
    else:
        grid_h = grid_w = int(np.sqrt(num_image_tokens)) if num_image_tokens > 1 else 24

    if os.path.exists(fn_attention):
        attentions = torch.load(fn_attention)
        logger.info(f'Loaded attention from {fn_attention}')
        if len(attentions) == len(state.output_ids_decoded):
            gr.Error('Mismatch between lengths of attentions and output tokens')
        
        num_tokens = len(attentions)
        num_layers = len(attentions[0])
        last_mh_attention = attentions[0][-1]
        batch_size, num_heads, inp_seq_len, seq_len = attentions[0][0].shape
        generated_text = state.output_ids_decoded
    
    else:
        return state, None
    
    # Img2TextAns Attention
    heatmap_mean = defaultdict(dict)
    if attn_modality_select == "Image-to-Answer":
        for layer_idx in range(1,num_layers):
            for head_idx in range(num_heads):
                mh_attentions = []
                mh_attentions = [attentions[i][layer_idx][:,:,-1,:].squeeze() for i in range(len(generated_text))]
                img_attn_list = []
                for mh_attention in mh_attentions:
                    img_end_idx = min(img_idx + num_image_tokens, mh_attention.shape[-1])
                    actual_tokens = img_end_idx - img_idx
                    if actual_tokens > 0:
                        attn_slice = mh_attention[head_idx, img_idx:img_end_idx].float().cpu()
                        # Reshape using actual grid dimensions
                        from scipy.ndimage import zoom
                        target_size = 24  # Standard output size
                        
                        if actual_tokens == num_image_tokens and actual_tokens == grid_h * grid_w:
                            # Reshape to actual grid dimensions
                            attn_grid = attn_slice.reshape(grid_h, grid_w).numpy()
                            # Resize to standard visualization size
                            zoom_factors = (target_size / grid_h, target_size / grid_w)
                            img_attn_list.append(torch.tensor(zoom(attn_grid, zoom_factors, order=1)))
                        else:
                            # Fallback: interpolate to standard grid
                            inferred_h = inferred_w = int(np.ceil(np.sqrt(actual_tokens)))
                            padded = torch.zeros(inferred_h * inferred_w)
                            padded[:actual_tokens] = attn_slice
                            padded = padded.reshape(inferred_h, inferred_w).numpy()
                            zoom_factors = (target_size / inferred_h, target_size / inferred_w)
                            img_attn_list.append(torch.tensor(zoom(padded, zoom_factors, order=1)))
                if img_attn_list:
                    img_attn = torch.stack(img_attn_list).numpy()
                    heatmap_mean[layer_idx][head_idx] = img_attn.mean()
                else:
                    heatmap_mean[layer_idx][head_idx] = 0.0
    elif attn_modality_select == "Question-to-Answer":
        fn_input_ids = state.attention_key + '_input_ids.pt'
        img_idx = state.image_idx
        input_ids = torch.load(fn_input_ids)
        len_question_only = input_ids.shape[1] - img_idx - num_image_tokens
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                mh_attentions = []
                mh_attentions = [attentions[i][layer_idx][:,:,-1,:].squeeze() for i in range(len(generated_text))]
                ques_start = img_idx + num_image_tokens
                ques_end = ques_start + len_question_only
                ques_attn = torch.stack([mh_attention[head_idx, ques_start:ques_end] for mh_attention in mh_attentions]).float().cpu().numpy()
                heatmap_mean[layer_idx][head_idx] = ques_attn.mean()
    heatmap_mean_df = pd.DataFrame(heatmap_mean)
    fig = plt.figure(figsize=(4, 4)) 
    ax = seaborn.heatmap(heatmap_mean_df,square=True, cbar_kws={"orientation": "horizontal"})
    ax.set_xlabel("Layers")
    ax.set_ylabel("Heads")
    ax.set_title(f"{attn_modality_select} Mean Attention")

    fig.tight_layout()
    return state, fig

def plot_text_to_image_analysis(state, layer_idx, boxes, head_idx=1 ):
    logger.info(f"plot_text_to_image_analysis called with layer={layer_idx}, head={head_idx}")

    if not hasattr(state, 'attention_key'):
        logger.warning("No attention_key found in state")
        return state, None, None
    
    img_recover = getattr(state, 'recovered_image', None)
    if img_recover is None:
        logger.warning("No recovered_image found - cannot plot attention")
        return state, None, None

    fn_attention = state.attention_key + '_attn.pt'
    img_idx = state.image_idx
    generated_text = state.output_ids_decoded
    
    # Get model-specific parameters
    num_image_tokens = getattr(state, 'num_image_tokens', 576)
    
    # Calculate grid dimensions
    if num_image_tokens == 576:
        grid_h, grid_w = 24, 24
    else:
        grid_h = grid_w = int(np.sqrt(num_image_tokens)) if num_image_tokens > 1 else 24

    # Sliders start at 1, convert to 0-indexed
    head_idx -= 1
    layer_idx -= 1
    
    # boxes is indexed as [row][col], extract selected patches as (row, col) tuples
    # This matches the attention tensor indexing of [head, token, row, col]
    img_patches = [(i, j) for i, row in enumerate(boxes) for j, clicked in enumerate(row) if clicked]
    logger.info(f"Selected patches from boxes: {img_patches}, boxes shape: {np.array(boxes).shape}, sum of selected: {np.sum(boxes)}")
    if len(img_patches) == 0:
        img_patches = [(5, 5)]
        logger.info("No patches selected, using default (5, 5)")
    if os.path.exists(fn_attention):
        attentions = torch.load(fn_attention)
        logger.info(f'Loaded attention from {fn_attention}')
        if len(attentions) == len(state.output_ids_decoded):
            gr.Error('Mismatch between lengths of attentions and output tokens')
        
        batch_size, num_heads, inp_seq_len, seq_len = attentions[0][0].shape
        
        # Clamp head_idx to valid range (0 to num_heads-1)
        head_idx = min(max(head_idx, 0), num_heads - 1)
        logger.info(f"Using head_idx={head_idx} (num_heads={num_heads})")
        
        generated_text = state.output_ids_decoded
    
    else:
        return state, None
    
    mh_attentions = []
    target_size = 24  # Standard output size for visualization
    for head_id in range(num_heads):
        att_per_head = []
        for i, out_att in enumerate(attentions):
            mh_attention = out_att[layer_idx]
            mh_attention = mh_attention[:, :, -1, :].unsqueeze(2)
            img_end_idx = min(img_idx + num_image_tokens, mh_attention.squeeze().shape[-1])
            actual_tokens = img_end_idx - img_idx
            
            if actual_tokens > 0:
                att_raw = mh_attention.squeeze()[head_id, img_idx:img_end_idx]
                from scipy.ndimage import zoom
                
                if actual_tokens == num_image_tokens and actual_tokens == grid_h * grid_w:
                    # Reshape to actual grid dimensions
                    att_grid = att_raw.reshape(grid_h, grid_w).cpu().float().numpy()
                    # Resize to standard visualization size
                    zoom_factors = (target_size / grid_h, target_size / grid_w)
                    att_img = torch.tensor(zoom(att_grid, zoom_factors, order=1))
                else:
                    # Fallback: interpolate to standard grid
                    inferred_h = inferred_w = int(np.ceil(np.sqrt(actual_tokens)))
                    padded = torch.zeros(inferred_h * inferred_w)
                    padded[:actual_tokens] = att_raw.float()
                    padded = padded.reshape(inferred_h, inferred_w).cpu().numpy()
                    zoom_factors = (target_size / inferred_h, target_size / inferred_w)
                    att_img = torch.tensor(zoom(padded, zoom_factors, order=1))
            else:
                att_img = torch.zeros((target_size, target_size))
            att_per_head.append(att_img)
        att_per_head = torch.stack(att_per_head)
        mh_attentions.append(att_per_head)
    mh_attentions = torch.stack(mh_attentions)

    img_mask = np.zeros((24, 24))
    for img_patch in img_patches:
        if img_patch[0] < 24 and img_patch[1] < 24:
            img_mask[img_patch[0], img_patch[1]] = 1
    img_mask = cmap(img_mask)
    img_mask = Image.fromarray((img_mask[:, :, :3] * 255 ).astype(np.uint8)).resize((336,336), Image.BICUBIC)
    img_mask.putalpha(208)
    img_patch_recovered = img_recover.copy()
    img_patch_recovered.paste(img_mask, mask=img_mask)
    img_patch_recovered

    words = generated_text
    # Filter valid patches within the 24x24 grid
    valid_patches = [(x, y) for x, y in img_patches if x < 24 and y < 24]
    if not valid_patches:
        valid_patches = [(5, 5)]
    float_values = torch.mean(torch.stack([mh_attentions[head_idx, :, x, y] for x, y in valid_patches]), dim=0).float().cpu()    
    normalized_values = (float_values - float_values.min()) / (float_values.max() - float_values.min() + 1e-8)

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])  # 2 columns, first column for the image, second column for the words
    ax_img = plt.subplot(gs[0])
    ax_img.imshow(img_patch_recovered)
    ax_img.axis('off')
    ax_words = plt.subplot(gs[1])
    x_position = 0.0

    for word, value in zip(words, normalized_values):
        color = plt.get_cmap("coolwarm")(value)
        color = to_rgba(color, alpha=0.6) 
        ax_words.text(x_position, 0.5, word, color=color, fontsize=14, ha='left', va='center')
        x_position += 0.10 

    cax = fig.add_axes([0.1, 0.15, 0.8, 0.03])  
    norm = plt.Normalize(min(normalized_values), max(normalized_values))
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([]) 
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Color Legend', labelpad=10, loc="center")

    ax_words.axis('off')
    plt.suptitle(f"Attention to the selected image patch(es) of head #{head_idx+1} and layer #{layer_idx+1}", fontsize=16, y=0.8, x=0.6)    

    attn_image_patch = torch.stack([mh_attentions[:, :, x, y] for x, y in valid_patches]).mean(0).float().cpu().mean(-1)
    logger.debug(torch.stack([mh_attentions[:, :, x, y] for x, y in valid_patches]).shape)
    logger.debug(torch.stack([mh_attentions[:, :, x, y] for x, y in valid_patches]).mean(0).shape)
    logger.debug(attn_image_patch.shape)
    
    fig2 = plt.figure(figsize=(10, 3))
    ax2 = seaborn.heatmap([attn_image_patch], 
        linewidths=.3, square=True, cbar_kws={"orientation": "horizontal", "shrink":0.3}
    )
    ax2.set_xlabel('Head number')
    ax2.set_title(f"Mean Head Attention between the image patches selected and the answer for layer {layer_idx+1}")
    fig2.tight_layout()
    return state, fig, fig2


def reset_tokens(state):
    generated_text = []
    for text in state.output_ids_decoded:
        generated_text.extend([(text, None), (' ', None)])

    return generated_text
    
