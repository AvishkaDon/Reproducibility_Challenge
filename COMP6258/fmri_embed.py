from MindVideo.models.fmri_encoder import fMRIEncoder
from MindVideo import create_Wen_dataset, create_Wen_test_data_only
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange
from typing import Callable, List, Optional, Union
from diffusers.utils import logging
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import torchvision
import cv2
import pickle

logger = logging.get_logger(__name__)

def load_clip_model():
    textModel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return textModel, tokenizer

def channel_first(img):
    if len(img.shape) == 3:
        if img.shape[0] == 3:
            return img
        img = rearrange(img, 'h w c -> c h w')
    elif len(img.shape) == 4:
        if img.shape[1] == 3:
            return img
        img = rearrange(img, 'f h w c -> f c h w')
    else:
        raise ValueError(f'img shape should be 3 or 4, but got {len(img.shape)}')
    return img

def normalize(img):
    if img.shape[-1] == 3 and len(img.shape) == 3:
        img = rearrange(img, 'h w c -> c h w')
    elif img.shape[-1] == 3 and len(img.shape) == 4:
        img = rearrange(img, 'f h w c -> f c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def pad_to_patch_size(x, patch_size):
    # pad the last dimension only
    padding_config = [(0,0)] * (x.ndim - 1) + [(0, patch_size-x.shape[-1]%patch_size)]
    return np.pad(x, padding_config, 'wrap')

def pad_to_length(x, length):
    if x.shape[-1] == length:
        return x
    # pad the last dimension only
    padding_config = [(0,0)] * (x.ndim - 1) + [(0, length-x.shape[-1])]
    return np.pad(x, padding_config, 'wrap')

def normalize_data(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def encode_fmri(fmri_encoder, fmri, device, num_videos_per_fmri, do_classifier_free_guidance, negative_prompt):
    dtype = fmri_encoder.dtype
    fmri_embeddings = fmri_encoder(fmri.to(device, dtype=dtype))
    # duplicate text embeddings for each generation per prompt
    bs_embed, seq_len, embed_dim = fmri_embeddings.shape
    fmri_embeddings = fmri_embeddings.repeat(1, num_videos_per_fmri, 1)
    fmri_embeddings = fmri_embeddings.view(bs_embed * num_videos_per_fmri, seq_len, -1)
    # support classification free guidance
    if do_classifier_free_guidance:
        uncond_input = negative_prompt.to(device, dtype=dtype)
        uncond_embeddings = fmri_encoder(uncond_input)
        # duplicate unconditional embeddings for each generation per prompt
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_fmri, 1)
        uncond_embeddings = uncond_embeddings.view(bs_embed * num_videos_per_fmri, seq_len, -1)
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        fmri_embeddings = torch.cat([uncond_embeddings, fmri_embeddings])

    return fmri_embeddings

def encode_prompt(prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt, tokenizer, text_encoder):
    batch_size = len(prompt) if isinstance(prompt, list) else 1

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer.model_max_length} tokens: {removed_text}"
        )

    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    text_embeddings = text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    text_embeddings = text_embeddings[0]

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
    text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        uncond_embeddings = uncond_embeddings[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings

# # Plot similarity matrix
def plot_similarity(similarity_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title(title)
    plt.show()

def expand_windows(fmri, idx, window_size):
    fmri = []
    t_idx = idx % t
    for i in range(window_size):
        fmri_to_return = fmri_patched[0][idx + i] if t_idx + i < t else fmri_patched[0][idx + (t - t_idx -1)]
        fmri.append(fmri_to_return)
    return torch.unsqueeze(torch.tensor(np.stack(fmri)), 0) # (window_size, num_voxels)


if __name__ == '__main__':
    # Set directory of Wen test set
    data_dir = './wen_testset'
    # Taken from eval_all_sub1.yaml
    val_data_setting = {
        'video_length': 6,
        'width': 256,
        'height': 256,
        'num_inference_steps': 200,
        'guidance_scale': 12.5,
        'num_videos_per_prompt': 1,
        'window_size': 2,
        'patch_size': 16,
        'half_precision': False,
        'subjects': ['subject1'],
        'eval_batch_size': 1
    }
    subjects = val_data_setting['subjects']
    dtype = torch.float16 if val_data_setting['half_precision'] else torch.float32
    window_size = val_data_setting['window_size']
    patch_size = val_data_setting['patch_size']
    w = val_data_setting['width']
    h = val_data_setting['height']

    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((w, h)), 
        channel_first
    ])
    fps = val_data_setting['video_length'] // 2
    dataset_test = create_Wen_test_data_only(data_dir, patch_size, 
                    fmri_transform=torch.FloatTensor, image_transform=[img_transform_test, img_transform_test], 
                    subjects=subjects, window_size=window_size, fps=fps)
    num_voxels = dataset_test.num_voxels
    eval_dataloader = DataLoader(dataset_test, batch_size=val_data_setting['eval_batch_size'], shuffle=False)

    model = fMRIEncoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fmri_encoder = fMRIEncoder.from_pretrained('./models', subfolder='fmri_encoder', num_voxels=num_voxels).to(device, dtype=dtype)
    fmri_encoder.eval()

    textModel, tokenizer = load_clip_model()
    textModel = textModel.to(device)

    num_videos_per_prompt = 1
    do_classifier_free_guidance = val_data_setting['guidance_scale'] > 1.0

    # Load video frames and segment IDs directly from the file paths
    wen_path = 'wen_testset/preprocessed/'
    video_frames_path = wen_path + 'video_test_256_3hz.npy'
    segment_ids_path = wen_path + 'test_seg_id_3hz.npy'
    text_ids_path = wen_path + 'text_test_256_3hz.npy'
    paths = ['fmri_test_subject1.npy', 'fmri_test_subject2.npy', 'fmri_test_subject3.npy']

    
    for i in range(3):
        fmri_ids_path = wen_path + paths[i]
        video_frames = np.load(video_frames_path)
        segment_ids = np.load(segment_ids_path)
        text_ids = np.load(text_ids_path)
        fmri_ids = np.load(fmri_ids_path)

        # Data Processing
        fmri_patched = pad_to_patch_size(normalize_data(fmri_ids, mean=0.0, std=1.0), patch_size)
        #t is video time - Might be wrong could be shape[1] * shape[2]
        t = video_frames.shape[2]

        fmri_embeddings = torch.empty((5, 240, 2, 77, 768), dtype=torch.float32)  # Specify dtype to torch.float32
        df_index = segment_ids.flatten()

        vid_num = 0
        for vid_num in range(fmri_ids.shape[0]):
            for idx in range(fmri_ids.shape[1]):
                uncon_fmri = np.mean(fmri_patched[vid_num, idx], axis=0, keepdims=True)    
                fmri_tensor = expand_windows(fmri_patched[vid_num, idx], idx, window_size).to(device)
                negative_prompt = expand_windows(uncon_fmri, idx, window_size).to(device)
                with torch.no_grad():
                    print(f"Encoding fMRI tensor with shape: {fmri_tensor.shape}")
                    fmri_embeddings[vid_num, idx] = encode_fmri(fmri_encoder, fmri_tensor, device, val_data_setting['num_videos_per_prompt'], do_classifier_free_guidance, negative_prompt).cpu()
                torch.cuda.empty_cache()  # Clear cached memory
                #break
    #print("DELETE BREAK ABOVE TO RUN FULL")
    print("done")
with open(f'fMRI_embeddings{i}.pkl', 'wb') as f:
    pickle.dump(fmri_embeddings, f)
raise ValueError("STOP")
