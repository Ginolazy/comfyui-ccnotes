## ComfyUI/custom_nodes/CCNotes/py/third-party.py
import comfy.utils
import comfy.model_management
import math
import nodes
import numpy as np
import torch
import kornia
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes
from .utility.type_utility import any_type

#-- comfyui-easy-use --#
class IfElse:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "boolean": ("BOOLEAN",),
                "on_true": (any_type, {"lazy": True}),
                "on_false": (any_type, {"lazy": True}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("*",)
    FUNCTION = "execute"
    CATEGORY = "CCNotes/Third-party/comfyui-easy-use"

    def check_lazy_status(self, boolean, on_true=None, on_false=None):
        if boolean and on_true is None:
            return ["on_true"]
        if not boolean and on_false is None:
            return ["on_false"]

    def execute(self, *args, **kwargs):
        return (kwargs['on_true'] if kwargs['boolean'] else kwargs['on_false'],)

class isMaskEmpty:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "CCNotes/Third-party/comfyui-easy-use"

    def execute(self, mask):
        if mask is None:
            return (True,)
        if torch.all(mask == 0):
            return (True,)
        return (False,)

class isNone:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "execute"
    CATEGORY = "CCNotes/Third-party/comfyui-easy-use"

    def execute(self, any):
        return (True if any is None else False,)

# comfyUI_essentials
class ImageColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_space": (["LAB", "YCbCr", "RGB", "LUV", "YUV", "XYZ"],),
                "factor": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, }),
                "device": (["auto", "cpu", "gpu"],),
                "batch_size": ("INT", { "default": 0, "min": 0, "max": 1024, "step": 1, }),
            },
            "optional": {
                "reference_mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CCNotes/Third-party/comfyUI_essentials"
    def execute(self, image, reference, color_space, factor, device, batch_size, reference_mask=None):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'
        image = image.permute([0, 3, 1, 2])
        reference = reference.permute([0, 3, 1, 2]).to(device)
        if reference_mask is not None:
            assert reference_mask.ndim == 3, f"Expected reference_mask to have 3 dimensions, but got {reference_mask.ndim}"
            assert reference_mask.shape[0] == reference.shape[0], f"Frame count mismatch: reference_mask has {reference_mask.shape[0]} frames, but reference has {reference.shape[0]}"
            reference_mask = reference_mask.unsqueeze(1).to(device)
            reference_mask = (reference_mask > 0.5).float()
            if reference_mask.shape[2:] != reference.shape[2:]:
                reference_mask = comfy.utils.common_upscale(
                    reference_mask,
                    reference.shape[3], reference.shape[2],
                    upscale_method='bicubic',
                    crop='center'
                )
        if batch_size == 0 or batch_size > image.shape[0]:
            batch_size = image.shape[0]
        if "LAB" == color_space:
            reference = kornia.color.rgb_to_lab(reference)
        elif "YCbCr" == color_space:
            reference = kornia.color.rgb_to_ycbcr(reference)
        elif "LUV" == color_space:
            reference = kornia.color.rgb_to_luv(reference)
        elif "YUV" == color_space:
            reference = kornia.color.rgb_to_yuv(reference)
        elif "XYZ" == color_space:
            reference = kornia.color.rgb_to_xyz(reference)
        reference_mean, reference_std = self.compute_mean_std(reference, reference_mask)
        image_batch = torch.split(image, batch_size, dim=0)
        output = []
        for image in image_batch:
            image = image.to(device)
            if color_space == "LAB":
                image = kornia.color.rgb_to_lab(image)
            elif color_space == "YCbCr":
                image = kornia.color.rgb_to_ycbcr(image)
            elif color_space == "LUV":
                image = kornia.color.rgb_to_luv(image)
            elif color_space == "YUV":
                image = kornia.color.rgb_to_yuv(image)
            elif color_space == "XYZ":
                image = kornia.color.rgb_to_xyz(image)
            image_mean, image_std = self.compute_mean_std(image)
            matched = torch.nan_to_num((image - image_mean) / image_std) * torch.nan_to_num(reference_std) + reference_mean
            matched = factor * matched + (1 - factor) * image
            if color_space == "LAB":
                matched = kornia.color.lab_to_rgb(matched)
            elif color_space == "YCbCr":
                matched = kornia.color.ycbcr_to_rgb(matched)
            elif color_space == "LUV":
                matched = kornia.color.luv_to_rgb(matched)
            elif color_space == "YUV":
                matched = kornia.color.yuv_to_rgb(matched)
            elif color_space == "XYZ":
                matched = kornia.color.xyz_to_rgb(matched)
            out = matched.permute([0, 2, 3, 1]).clamp(0, 1).to(comfy.model_management.intermediate_device())
            output.append(out)
        out = None
        output = torch.cat(output, dim=0)
        return (output,)
    def compute_mean_std(self, tensor, mask=None):
        if mask is not None:
            masked_tensor = tensor * mask
            mask_sum = mask.sum(dim=[2, 3], keepdim=True)
            mask_sum = torch.clamp(mask_sum, min=1e-6)
            mean = torch.nan_to_num(masked_tensor.sum(dim=[2, 3], keepdim=True) / mask_sum)
            std = torch.sqrt(torch.nan_to_num(((masked_tensor - mean) ** 2 * mask).sum(dim=[2, 3], keepdim=True) / mask_sum))
        else:
            mean = tensor.mean(dim=[2, 3], keepdim=True)
            std = tensor.std(dim=[2, 3], keepdim=True)
        return mean, std

class ImageColorMatchAdobe(ImageColorMatch):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_space": (["RGB", "LAB"],),
                "luminance_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "color_intensity_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "fade_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "neutralization_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "device": (["auto", "cpu", "gpu"],),
            },
            "optional": {
                "reference_mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "CCNotes/Third-party/comfyUI_essentials"
    def analyze_color_statistics(self, image, mask=None):
        l, a, b = kornia.color.rgb_to_lab(image).chunk(3, dim=1)
        if mask is not None:
            mask = F.interpolate(mask, size=image.shape[2:], mode='nearest')
            mask = (mask > 0.5).float()
            l = l * mask
            a = a * mask
            b = b * mask
            num_pixels = mask.sum()
            mean_l = (l * mask).sum() / num_pixels
            mean_a = (a * mask).sum() / num_pixels
            mean_b = (b * mask).sum() / num_pixels
            std_l = torch.sqrt(((l - mean_l)**2 * mask).sum() / num_pixels)
            var_ab = ((a - mean_a)**2 + (b - mean_b)**2) * mask
            std_ab = torch.sqrt(var_ab.sum() / num_pixels)
        else:
            mean_l = l.mean()
            std_l = l.std()
            mean_a = a.mean()
            mean_b = b.mean()
            std_ab = torch.sqrt(a.var() + b.var())
        return mean_l, std_l, mean_a, mean_b, std_ab
    def apply_color_transformation(self, image, source_stats, dest_stats, L, C, N):
        l, a, b = kornia.color.rgb_to_lab(image).chunk(3, dim=1)
        src_mean_l, src_std_l, src_mean_a, src_mean_b, src_std_ab = source_stats
        dest_mean_l, dest_std_l, dest_mean_a, dest_mean_b, dest_std_ab = dest_stats
        l_new = (l - dest_mean_l) * (src_std_l / dest_std_l) * L + src_mean_l
        a = a - N * dest_mean_a
        b = b - N * dest_mean_b
        a_new = a * (src_std_ab / dest_std_ab) * C
        b_new = b * (src_std_ab / dest_std_ab) * C
        lab_new = torch.cat([l_new, a_new, b_new], dim=1)
        rgb_new = kornia.color.lab_to_rgb(lab_new)
        return rgb_new
    def execute(self, image, reference, color_space, luminance_factor, color_intensity_factor, fade_factor, neutralization_factor, device, reference_mask=None):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'
        image = image.permute(0, 3, 1, 2).to(device)
        reference = reference.permute(0, 3, 1, 2).to(device)
        if reference_mask is not None:
            if reference_mask.ndim == 2:
                reference_mask = reference_mask.unsqueeze(0).unsqueeze(0)
            elif reference_mask.ndim == 3:
                reference_mask = reference_mask.unsqueeze(1)
            reference_mask = reference_mask.to(device)
        source_stats = self.analyze_color_statistics(reference, reference_mask)
        dest_stats = self.analyze_color_statistics(image)
        transformed = self.apply_color_transformation(
            image, source_stats, dest_stats, 
            luminance_factor, color_intensity_factor, neutralization_factor
        )
        result = fade_factor * transformed + (1 - fade_factor) * image
        result = result.permute(0, 2, 3, 1).clamp(0, 1).to(comfy.model_management.intermediate_device())
        return (result,)

#-- comfyui-inpaint-cropandstitch --#
def rescale(samples, width, height, algorithm: str):
    if algorithm == "bislerp":  # convert for compatibility with old workflows
        algorithm = "bicubic"
    algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    return samples
def rescale_i(samples, width, height, algorithm: str):
    samples = samples.movedim(-1, 1)
    algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    samples = samples.movedim(1, -1)
    return samples
def rescale_m(samples, width, height, algorithm: str):
    samples = samples.unsqueeze(1)
    algorithm = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
    samples_pil: Image.Image = F.to_pil_image(samples[0].cpu()).resize((width, height), algorithm)
    samples = F.to_tensor(samples_pil).unsqueeze(0)
    samples = samples.squeeze(1)
    return samples
def preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
    current_width, current_height = image.shape[2], image.shape[1]  # Image size [batch, height, width, channels]
    if preresize_mode == "ensure minimum resolution":
        if current_width >= preresize_min_width and current_height >= preresize_min_height:
            return image, mask, optional_context_mask
        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor = max(scale_factor_min_width, scale_factor_min_height)
        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)
        image = rescale_i(image, target_width, target_height, upscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
        assert target_width >= preresize_min_width and target_height >= preresize_min_height, \
            f"Internal error: After resizing, target size {target_width}x{target_height} is smaller than min size {preresize_min_width}x{preresize_min_height}"
    elif preresize_mode == "ensure minimum and maximum resolution":
        if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
            return image, mask, optional_context_mask
        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)
        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)
        if scale_factor_min > 1 and scale_factor_max < 1:
            assert False, "Cannot meet both minimum and maximum resolution requirements with aspect ratio preservation."
        if scale_factor_min > 1:  # We're upscaling to meet min resolution
            scale_factor = scale_factor_min
            rescale_algorithm = upscale_algorithm  # Use upscale algorithm for min resolution
        else:  # We're downscaling to meet max resolution
            scale_factor = scale_factor_max
            rescale_algorithm = downscale_algorithm  # Use downscale algorithm for max resolution
        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)
        image = rescale_i(image, target_width, target_height, rescale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        assert preresize_min_width <= target_width <= preresize_max_width, \
            f"Internal error: Target width {target_width} is outside the range {preresize_min_width} - {preresize_max_width}"
        assert preresize_min_height <= target_height <= preresize_max_height, \
            f"Internal error: Target height {target_height} is outside the range {preresize_min_height} - {preresize_max_height}"
    elif preresize_mode == "ensure maximum resolution":
        if current_width <= preresize_max_width and current_height <= preresize_max_height:
            return image, mask, optional_context_mask
        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)
        target_width = int(current_width * scale_factor_max)
        target_height = int(current_height * scale_factor_max)
        image = rescale_i(image, target_width, target_height, downscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest')  # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest')  # Always nearest for efficiency
        assert target_width <= preresize_max_width and target_height <= preresize_max_height, \
            f"Internal error: Target size {target_width}x{target_height} is greater than max size {preresize_max_width}x{preresize_max_height}"
    return image, mask, optional_context_mask
def fillholes_iterative_hipass_fill_m(samples):
    thresholds = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    mask_np = samples.squeeze(0).cpu().numpy()
    for threshold in thresholds:
        thresholded_mask = mask_np >= threshold
        closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
        filled_mask = binary_fill_holes(closed_mask)
        mask_np = np.maximum(mask_np, np.where(filled_mask != 0, threshold, 0))
    final_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
    return final_mask
def hipassfilter_m(samples, threshold):
    filtered_mask = samples.clone()
    filtered_mask[filtered_mask < threshold] = 0
    return filtered_mask
def expand_m(mask, pixels):
    sigma = pixels / 4
    mask_np = mask.squeeze(0).cpu().numpy()
    kernel_size = math.ceil(sigma * 1.5 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated_mask = grey_dilation(mask_np, footprint=kernel)
    dilated_mask = dilated_mask.astype(np.float32)
    dilated_mask = torch.from_numpy(dilated_mask)
    dilated_mask = torch.clamp(dilated_mask, 0.0, 1.0)
    return dilated_mask.unsqueeze(0)
def invert_m(samples):
    inverted_mask = samples.clone()
    inverted_mask = 1.0 - inverted_mask
    return inverted_mask
def blur_m(samples, pixels):
    mask = samples.squeeze(0)
    sigma = pixels / 4 
    mask_np = mask.cpu().numpy()
    blurred_mask = gaussian_filter(mask_np, sigma=sigma)
    blurred_mask = torch.from_numpy(blurred_mask).float()
    blurred_mask = torch.clamp(blurred_mask, 0.0, 1.0)
    return blurred_mask.unsqueeze(0)
def extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
    B, H, W, C = image.shape
    new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
    new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))
    assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
    assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"
    expanded_image = torch.zeros(1, new_H, new_W, C, device=image.device)
    expanded_mask = torch.ones(1, new_H, new_W, device=mask.device)
    expanded_optional_context_mask = torch.zeros(1, new_H, new_W, device=optional_context_mask.device)
    up_padding = int(H * (extend_up_factor - 1.0))
    down_padding = new_H - H - up_padding
    left_padding = int(W * (extend_left_factor - 1.0))
    right_padding = new_W - W - left_padding
    slice_target_up = max(0, up_padding)
    slice_target_down = min(new_H, up_padding + H)
    slice_target_left = max(0, left_padding)
    slice_target_right = min(new_W, left_padding + W)
    slice_source_up = max(0, -up_padding)
    slice_source_down = min(H, new_H - up_padding)
    slice_source_left = max(0, -left_padding)
    slice_source_right = min(W, new_W - left_padding)
    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image[:, :, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = image[:, :, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    if up_padding > 0:
        expanded_image[:, :, :up_padding, slice_target_left:slice_target_right] = image[:, :, 0:1, slice_source_left:slice_source_right].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, slice_target_left:slice_target_right] = image[:, :, -1:, slice_source_left:slice_source_right].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, :, :left_padding] = expanded_image[:, :, :, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, :, -right_padding:] = expanded_image[:, :, :, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)
    expanded_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    expanded_optional_context_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = optional_context_mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    return expanded_image, expanded_mask, expanded_optional_context_mask
def findcontextarea_m(mask):
    mask_squeezed = mask[0]  # Now shape is [H, W]
    non_zero_indices = torch.nonzero(mask_squeezed)
    H, W = mask_squeezed.shape
    if non_zero_indices.numel() == 0:
        x, y = -1, -1
        w, h = -1, -1
    else:
        y = torch.min(non_zero_indices[:, 0]).item()
        x = torch.min(non_zero_indices[:, 1]).item()
        y_max = torch.max(non_zero_indices[:, 0]).item()
        x_max = torch.max(non_zero_indices[:, 1]).item()
        w = x_max - x + 1  # +1 to include the max index
        h = y_max - y + 1  # +1 to include the max index
    context = mask[:, y:y+h, x:x+w]
    return context, x, y, w, h
def growcontextarea_m(context, mask, x, y, w, h, extend_factor):
    img_h, img_w = mask.shape[1], mask.shape[2]
    grow_left = int(round(w * (extend_factor-1.0) / 2.0))    # Compute intended growth in each direction
    grow_right = int(round(w * (extend_factor-1.0) / 2.0))
    grow_up = int(round(h * (extend_factor-1.0) / 2.0))
    grow_down = int(round(h * (extend_factor-1.0) / 2.0))
    new_x = x - grow_left    # Try to grow left, but clamp at 0
    if new_x < 0:
        new_x = 0
    new_y = y - grow_up    # Try to grow up, but clamp at 0
    if new_y < 0:
        new_y = 0
    new_x2 = x + w + grow_right    # Right edge
    if new_x2 > img_w:
        new_x2 = img_w
    new_y2 = y + h + grow_down    # Bottom edge
    if new_y2 > img_h:
        new_y2 = img_h
    new_w = new_x2 - new_x    # New width and height
    new_h = new_y2 - new_y
    new_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]    # Extract the context
    if new_h < 0 or new_w < 0:
        new_x = 0
        new_y = 0
        new_w = mask.shape[2]
        new_h = mask.shape[1]
    return new_context, new_x, new_y, new_w, new_h
def combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask):
    _, x_opt, y_opt, w_opt, h_opt = findcontextarea_m(optional_context_mask)
    if x == -1:
        x, y, w, h = x_opt, y_opt, w_opt, h_opt
    if x_opt == -1:
        x_opt, y_opt, w_opt, h_opt = x, y, w, h
    if x == -1:
        return torch.zeros(1, 0, 0, device=mask.device), -1, -1, -1, -1
    new_x = min(x, x_opt)
    new_y = min(y, y_opt)
    new_x_max = max(x + w, x_opt + w_opt)
    new_y_max = max(y + h, y_opt + h_opt)
    new_w = new_x_max - new_x
    new_h = new_y_max - new_y
    combined_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]
    return combined_context, new_x, new_y, new_w, new_h
def pad_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)
def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm):
    image = image.clone()
    mask = mask.clone()
    # Ok this is the most complex function in this node. The one that does the magic after all the preparation done by the other nodes.
    # Basically this function determines the right context area that encompasses the whole context area (mask+optional_context_mask),
    # that is ideally within the bounds of the original image, and that has the right aspect ratio to match target width and height.
    # It may grow the image if the aspect ratio wouldn't fit in the original image.
    # It keeps track of that growing to then be able to crop the image in the stitch node.
    # Finally, it crops the context area and resizes it to be exactly target_w and target_h.
    # It keeps track of that resize to be able to revert it in the stitch node.
    # Check for invalid inputs
    if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]
    # Step 1: Pad target dimensions to be multiples of padding
    if padding != 0:
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)
    # Step 2: Calculate target aspect ratio
    target_aspect_ratio = target_w / target_h
    # Step 3: Grow current context area to meet the target aspect ratio
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h
    if context_aspect_ratio < target_aspect_ratio:
        new_w = int(h * target_aspect_ratio)        # Grow width to meet aspect ratio
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y
        if new_x < 0:        # Adjust new_x to keep within bounds
            shift = -new_x
            if new_x + new_w + shift <= image_w:
                new_x += shift
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow
        elif new_x + new_w > image_w:
            overflow = new_x + new_w - image_w
            if new_x - overflow >= 0:
                new_x -= overflow
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow
    else:
        new_w = w        # Grow height to meet aspect ratio
        new_h = int(w / target_aspect_ratio)
        new_x = x
        new_y = y - (new_h - h) // 2
        if new_y < 0:        # Adjust new_y to keep within bounds
            shift = -new_y
            if new_y + new_h + shift <= image_h:
                new_y += shift
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
        elif new_y + new_h > image_h:
            overflow = new_y + new_h - image_h
            if new_y - overflow >= 0:
                new_y -= overflow
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
    # Step 4: Grow the image to accommodate the new context area
    up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0
    expanded_image_w = image_w
    expanded_image_h = image_h
    # Adjust width for left overflow (x < 0) and right overflow (x + w > image_w)
    if new_x < 0:
        left_padding = -new_x
        expanded_image_w += left_padding
    if new_x + new_w > image_w:
        right_padding = (new_x + new_w - image_w)
        expanded_image_w += right_padding
    # Adjust height for top overflow (y < 0) and bottom overflow (y + h > image_h)
    if new_y < 0:
        up_padding = -new_y
        expanded_image_h += up_padding 
    if new_y + new_h > image_h:
        down_padding = (new_y + new_h - image_h)
        expanded_image_h += down_padding
    # Step 5: Create the new image and mask
    expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
    expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)
    # Reorder the tensors to match the required dimension format for padding
    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    # Ensure the expanded image has enough room to hold the padded version of the original image
    expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image
    # Fill the new extended areas with the edge values of the image
    if up_padding > 0:
        expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = image[:, :, 0:1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = image[:, :, -1:, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)
    # Reorder the tensors back to [B, H, W, C] format
    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    # Same for the mask
    expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask
    # Record the cto values (canvas to original)
    cto_x = left_padding
    cto_y = up_padding
    cto_w = image_w
    cto_h = image_h
    # The final expanded image and mask
    canvas_image = expanded_image
    canvas_mask = expanded_mask
    # Step 6: Crop the image and mask around x, y, w, h
    ctc_x = new_x+left_padding
    ctc_y = new_y+up_padding
    ctc_w = new_w
    ctc_h = new_h
    # Crop the image and mask
    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    # Step 7: Resize image and mask to the target width and height
    # Decide which algorithm to use based on the scaling direction
    if target_w > ctc_w or target_h > ctc_h:  # Upscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
    else:  # Downscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)
    return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h
def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    canvas_image = canvas_image.clone()
    inpainted_image = inpainted_image.clone()
    mask = mask.clone()
    # Resize inpainted image and mask to match the context size
    _, h, w, _ = inpainted_image.shape
    if ctc_w > w or ctc_h > h:  # Upscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
    else:  # Downscaling
        resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
        resized_mask = rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)
    # Clamp mask to [0, 1] and expand to match image channels
    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # shape: [1, H, W, 1]
    # Extract the canvas region we're about to overwrite
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
    # Blend: new = mask * inpainted + (1 - mask) * canvas
    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
    # Paste the blended region back onto the canvas
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended
    # Final crop to get back the original image area
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]
    return output_image
class InpaintCrop:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

    This node crop before sampling and stitch after sampling for fast, efficient inpainting without altering unmasked areas.
    Context area can be specified via expand pixels and expand factor or via a separate (optional) mask.
    Works free size, forced size, and ranged size.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "context_expand_pixels": ("INT", {"default": 20, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "context_expand_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01}),
                "fill_mask_holes": ("BOOLEAN", {"default": True}),
                "blur_mask_pixels": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 256.0, "step": 0.1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "blend_pixels": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "rescale_algorithm": (["nearest", "bilinear", "bicubic", "bislerp", "lanczos", "box", "hamming"], {"default": "bicubic"}),
                "mode": (["ranged size", "forced size", "free size"], {"default": "ranged size"}),
                "force_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}), # force
                "force_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}), # force
                "rescale_factor": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100.0, "step": 0.01}), # free
                "min_width": ("INT", {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}), # ranged
                "min_height": ("INT", {"default": 512, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}), # ranged
                "max_width": ("INT", {"default": 768, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}), # ranged
                "max_height": ("INT", {"default": 768, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}), # ranged
                "padding": ([8, 16, 32, 64, 128, 256, 512], {"default": 32}), # free and ranged
           },
           "optional": {
                "optional_context_mask": ("MASK",),
           }
        }
    CATEGORY = "CCNotes/Third-party/comfyui-inpaint-cropandstitch"
    RETURN_TYPES = ("STITCH", "IMAGE", "MASK")
    RETURN_NAMES = ("stitch", "cropped_image", "cropped_mask")
    FUNCTION = "inpaint_crop"
    def grow_and_blur_mask(self, mask, blur_pixels):
        if blur_pixels > 0.001:
            sigma = blur_pixels / 4
            growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in growmask:
                mask_np = m.numpy()
                kernel_size = math.ceil(sigma * 1.5 + 1)
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                dilated_mask = grey_dilation(mask_np, footprint=kernel)
                output = dilated_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)
            mask_np = mask.numpy()
            filtered_mask = gaussian_filter(mask_np, sigma=sigma)
            mask = torch.from_numpy(filtered_mask)
            mask = torch.clamp(mask, 0.0, 1.0)
        return mask
    def adjust_to_aspect_ratio(self, x_min, x_max, y_min, y_max, width, height, target_width, target_height):
        x_min_key, x_max_key, y_min_key, y_max_key = x_min, x_max, y_min, y_max
        # Calculate the current width and height
        current_width = x_max - x_min + 1
        current_height = y_max - y_min + 1
        # Calculate aspect ratios
        aspect_ratio = target_width / target_height
        current_aspect_ratio = current_width / current_height
        if current_aspect_ratio < aspect_ratio:
            # Adjust width to match target aspect ratio
            new_width = int(current_height * aspect_ratio)
            extend_x = (new_width - current_width)
            x_min = max(x_min - extend_x//2, 0)
            x_max = min(x_max + extend_x//2, width - 1)
        else:
            # Adjust height to match target aspect ratio
            new_height = int(current_width / aspect_ratio)
            extend_y = (new_height - current_height)
            y_min = max(y_min - extend_y//2, 0)
            y_max = min(y_max + extend_y//2, height - 1)
        return int(x_min), int(x_max), int(y_min), int(y_max)
    def adjust_to_preferred(self, x_min, x_max, y_min, y_max, width, height, preferred_x_start, preferred_x_end, preferred_y_start, preferred_y_end):
        # Ensure the area is within preferred bounds as much as possible
        if preferred_x_start <= x_min and preferred_x_end >= x_max and preferred_y_start <= y_min and preferred_y_end >= y_max:
            return x_min, x_max, y_min, y_max
        # Shift x_min and x_max to fit within preferred bounds if possible
        if x_max - x_min + 1 <= preferred_x_end - preferred_x_start + 1:
            if x_min < preferred_x_start:
                x_shift = preferred_x_start - x_min
                x_min += x_shift
                x_max += x_shift
            elif x_max > preferred_x_end:
                x_shift = x_max - preferred_x_end
                x_min -= x_shift
                x_max -= x_shift
        # Shift y_min and y_max to fit within preferred bounds if possible
        if y_max - y_min + 1 <= preferred_y_end - preferred_y_start + 1:
            if y_min < preferred_y_start:
                y_shift = preferred_y_start - y_min
                y_min += y_shift
                y_max += y_shift
            elif y_max > preferred_y_end:
                y_shift = y_max - preferred_y_end
                y_min -= y_shift
                y_max -= y_shift
        return int(x_min), int(x_max), int(y_min), int(y_max)
    def apply_padding(self, min_val, max_val, max_boundary, padding):
        # Calculate the midpoint and the original range size
        original_range_size = max_val - min_val + 1
        midpoint = (min_val + max_val) // 2
        # Determine the smallest multiple of padding that is >= original_range_size
        if original_range_size % padding == 0:
            new_range_size = original_range_size
        else:
            new_range_size = (original_range_size // padding + 1) * padding
        # Calculate the new min and max values centered on the midpoint
        new_min_val = max(midpoint - new_range_size // 2, 0)
        new_max_val = new_min_val + new_range_size - 1
        # Ensure the new max doesn't exceed the boundary
        if new_max_val >= max_boundary:
            new_max_val = max_boundary - 1
            new_min_val = max(new_max_val - new_range_size + 1, 0)
        # Ensure the range still ends on a multiple of padding
        # Adjust if the calculated range isn't feasible within the given constraints
        if (new_max_val - new_min_val + 1) != new_range_size:
            new_min_val = max(new_max_val - new_range_size + 1, 0)
        return new_min_val, new_max_val
    def inpaint_crop(self, image, mask, context_expand_pixels, context_expand_factor, fill_mask_holes, blur_mask_pixels, invert_mask, blend_pixels, mode, rescale_algorithm, force_width, force_height, rescale_factor, padding, min_width, min_height, max_width, max_height, optional_context_mask=None):
        if image.shape[0] > 1:
            assert mode == "forced size", "Mode must be 'forced size' when input is a batch of images"
        assert image.shape[0] == mask.shape[0], "Batch size of images and masks must be the same"
        if optional_context_mask is not None:
            assert optional_context_mask.shape[0] == image.shape[0], "Batch size of optional_context_masks must be the same as images or None"
        result_stitch = {'x': [], 'y': [], 'original_image': [], 'cropped_mask_blend': [], 'rescale_x': [], 'rescale_y': [], 'start_x': [], 'start_y': [], 'initial_width': [], 'initial_height': []}
        results_image = []
        results_mask = []
        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = None
            if optional_context_mask is not None:
                one_optional_context_mask = optional_context_mask[b].unsqueeze(0)
            stitch, cropped_image, cropped_mask = self.inpaint_crop_single_image(one_image, one_mask, context_expand_pixels, context_expand_factor, fill_mask_holes, blur_mask_pixels, invert_mask, blend_pixels, mode, rescale_algorithm, force_width, force_height, rescale_factor, padding, min_width, min_height, max_width, max_height, one_optional_context_mask)
            for key in result_stitch:
                result_stitch[key].append(stitch[key])
            cropped_image = cropped_image.squeeze(0)
            results_image.append(cropped_image)
            cropped_mask = cropped_mask.squeeze(0)
            results_mask.append(cropped_mask)
        result_image = torch.stack(results_image, dim=0)
        result_mask = torch.stack(results_mask, dim=0)
        return result_stitch, result_image, result_mask
    # Parts of this function are from KJNodes: https://github.com/kijai/ComfyUI-KJNodes
    def inpaint_crop_single_image(self, image, mask, context_expand_pixels, context_expand_factor, fill_mask_holes, blur_mask_pixels, invert_mask, blend_pixels, mode, rescale_algorithm, force_width, force_height, rescale_factor, padding, min_width, min_height, max_width, max_height, optional_context_mask=None):
        #Validate or initialize mask
        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                mask = torch.zeros_like(image[:, :, :, 0])
            else:
                assert False, "mask size must match image size"
        # Fill holes if requested
        if fill_mask_holes:
            holemask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
            out = []
            for m in holemask:
                mask_np = m.numpy()
                binary_mask = mask_np > 0
                struct = np.ones((5, 5))
                closed_mask = binary_closing(binary_mask, structure=struct, border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                output = filled_mask.astype(np.float32) * 255
                output = torch.from_numpy(output)
                out.append(output)
            mask = torch.stack(out, dim=0)
            mask = torch.clamp(mask, 0.0, 1.0)
        # Grow and blur mask if requested
        if blur_mask_pixels > 0.001:
            mask = self.grow_and_blur_mask(mask, blur_mask_pixels)
        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask
        # Validate or initialize context mask
        if optional_context_mask is None:
            context_mask = mask
        elif optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
            non_zero_indices = torch.nonzero(optional_context_mask[0], as_tuple=True)
            if not non_zero_indices[0].size(0):
                context_mask = mask
            else:
                assert False, "context_mask size must match image size"
        else:
            context_mask = optional_context_mask + mask 
            context_mask = torch.clamp(context_mask, 0.0, 1.0)
        # Ensure mask dimensions match image dimensions except channels
        initial_batch, initial_height, initial_width, initial_channels = image.shape
        mask_batch, mask_height, mask_width = mask.shape
        context_mask_batch, context_mask_height, context_mask_width = context_mask.shape
        assert initial_height == mask_height and initial_width == mask_width, "Image and mask dimensions must match"
        assert initial_height == context_mask_height and initial_width == context_mask_width, "Image and context mask dimensions must match"
        # Extend image and masks to turn it into a big square in case the context area would go off bounds
        extend_y = (initial_width + 1) // 2 # Intended, extend height by width (turn into square)
        extend_x = (initial_height + 1) // 2 # Intended, extend width by height (turn into square)
        new_height = initial_height + 2 * extend_y
        new_width = initial_width + 2 * extend_x
        start_y = extend_y
        start_x = extend_x
        available_top = min(start_y, initial_height)
        available_bottom = min(new_height - (start_y + initial_height), initial_height)
        available_left = min(start_x, initial_width)
        available_right = min(new_width - (start_x + initial_width), initial_width)
        new_image = torch.zeros((initial_batch, new_height, new_width, initial_channels), dtype=image.dtype)
        new_image[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :] = image
        # Mirror image so there's no bleeding of black border when using inpaintmodelconditioning
        # Top
        new_image[:, start_y - available_top:start_y, start_x:start_x + initial_width, :] = torch.flip(image[:, :available_top, :, :], [1])
        # Bottom
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width, :] = torch.flip(image[:, -available_bottom:, :, :], [1])
        # Left
        new_image[:, start_y:start_y + initial_height, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x:start_x + available_left, :], [2])
        # Right
        new_image[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [2])
        # Top-left corner
        new_image[:, start_y - available_top:start_y, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x:start_x + available_left, :], [1, 2])
        # Top-right corner
        new_image[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])
        # Bottom-left corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left, :], [1, 2])
        # Bottom-right corner
        new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])
        new_mask = torch.ones((mask_batch, new_height, new_width), dtype=mask.dtype) # assume ones in extended image
        new_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask
        blend_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype) # assume zeros in extended image
        blend_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = mask
        # Mirror blend mask so there's no bleeding of border when blending
        # Top
        blend_mask[:, start_y - available_top:start_y, start_x:start_x + initial_width] = torch.flip(mask[:, :available_top, :], [1])
        # Bottom
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width] = torch.flip(mask[:, -available_bottom:, :], [1])
        # Left
        blend_mask[:, start_y:start_y + initial_height, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y:start_y + initial_height, start_x:start_x + available_left], [2])
        # Right
        blend_mask[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width], [2])
        # Top-left corner
        blend_mask[:, start_y - available_top:start_y, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y:start_y + available_top, start_x:start_x + available_left], [1, 2])
        # Top-right corner
        blend_mask[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width], [1, 2])
        # Bottom-left corner
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x] = torch.flip(blend_mask[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left], [1, 2])
        # Bottom-right corner
        blend_mask[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right] = torch.flip(blend_mask[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width], [1, 2])
        new_context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=context_mask.dtype)
        new_context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] = context_mask
        image = new_image
        mask = new_mask
        context_mask = new_context_mask
        original_image = image
        original_mask = mask
        original_width = image.shape[2]
        original_height = image.shape[1]
        # If there are no non-zero indices in the context_mask, adjust context mask to the whole image
        non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        if not non_zero_indices[0].size(0):
            context_mask = torch.ones_like(image[:, :, :, 0])
            context_mask = torch.zeros((mask_batch, new_height, new_width), dtype=mask.dtype)
            context_mask[:, start_y:start_y + initial_height, start_x:start_x + initial_width] += 1.0
            non_zero_indices = torch.nonzero(context_mask[0], as_tuple=True)
        # Compute context area from context mask
        y_min = torch.min(non_zero_indices[0]).item()
        y_max = torch.max(non_zero_indices[0]).item()
        x_min = torch.min(non_zero_indices[1]).item()
        x_max = torch.max(non_zero_indices[1]).item()
        height = context_mask.shape[1]
        width = context_mask.shape[2]
        # Grow context area if requested
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        y_grow = round(max(y_size*(context_expand_factor-1), context_expand_pixels, blend_pixels**1.5))
        x_grow = round(max(x_size*(context_expand_factor-1), context_expand_pixels, blend_pixels**1.5))
        y_min = max(y_min - y_grow // 2, 0)
        y_max = min(y_max + y_grow // 2, height - 1)
        x_min = max(x_min - x_grow // 2, 0)
        x_max = min(x_max + x_grow // 2, width - 1)
        y_size = y_max - y_min + 1
        x_size = x_max - x_min + 1
        effective_upscale_factor_x = 1.0
        effective_upscale_factor_y = 1.0
        # Adjust to preferred size
        if mode == 'forced size':
            #Sub case of ranged size.
            min_width = max_width = force_width
            min_height = max_height = force_height
        if mode == 'ranged size' or mode == 'forced size':
            assert max_width >= min_width, "max_width must be greater than or equal to min_width"
            assert max_height >= min_height, "max_height must be greater than or equal to min_height"
            # Ensure we set an aspect ratio supported by min_width, max_width, min_height, max_height
            current_width = x_max - x_min + 1
            current_height = y_max - y_min + 1
            # Calculate aspect ratio of the selected area
            current_aspect_ratio = current_width / current_height
            # Calculate the aspect ratio bounds
            min_aspect_ratio = min_width / max_height
            max_aspect_ratio = max_width / min_height
            # Adjust target width and height based on aspect ratio bounds
            if current_aspect_ratio < min_aspect_ratio:
                # Adjust to meet minimum width constraint
                target_width = min(current_width, min_width)
                target_height = int(target_width / min_aspect_ratio)
                x_min, x_max, y_min, y_max = self.adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height)
                x_min, x_max, y_min, y_max = self.adjust_to_preferred(x_min, x_max, y_min, y_max, width, height, start_x, start_x+initial_width, start_y, start_y+initial_height)
            elif current_aspect_ratio > max_aspect_ratio:
                # Adjust to meet maximum width constraint
                target_height = min(current_height, max_height)
                target_width = int(target_height * max_aspect_ratio)
                x_min, x_max, y_min, y_max = self.adjust_to_aspect_ratio(x_min, x_max, y_min, y_max, width, height, target_width, target_height)
                x_min, x_max, y_min, y_max = self.adjust_to_preferred(x_min, x_max, y_min, y_max, width, height, start_x, start_x+initial_width, start_y, start_y+initial_height)
            else:
                # Aspect ratio is within bounds, keep the current size
                target_width = current_width
                target_height = current_height
            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1
            # Adjust to min and max sizes
            max_rescale_width = max_width / x_size
            max_rescale_height = max_height / y_size
            max_rescale_factor = min(max_rescale_width, max_rescale_height)
            rescale_factor = max_rescale_factor
            min_rescale_width = min_width / x_size
            min_rescale_height = min_height / y_size
            min_rescale_factor = min(min_rescale_width, min_rescale_height)
            rescale_factor = max(min_rescale_factor, rescale_factor)
        # Upscale image and masks if requested, they will be downsized at stitch phase
        if rescale_factor < 0.999 or rescale_factor > 1.001:
            samples = image            
            samples = samples.movedim(-1, 1)
            width = round(samples.shape[3] * rescale_factor)
            height = round(samples.shape[2] * rescale_factor)
            samples = rescale(samples, width, height, rescale_algorithm)
            effective_upscale_factor_x = float(width)/float(original_width)
            effective_upscale_factor_y = float(height)/float(original_height)
            samples = samples.movedim(1, -1)
            image = samples
            samples = mask
            samples = samples.unsqueeze(1)
            samples = rescale(samples, width, height, "nearest")
            samples = samples.squeeze(1)
            mask = samples
            samples = blend_mask
            samples = samples.unsqueeze(1)
            samples = rescale(samples, width, height, "nearest")
            samples = samples.squeeze(1)
            blend_mask = samples
            # Do math based on min,size instead of min,max to avoid rounding errors
            y_size = y_max - y_min + 1
            x_size = x_max - x_min + 1
            target_x_size = int(x_size * effective_upscale_factor_x)
            target_y_size = int(y_size * effective_upscale_factor_y)
            x_min = round(x_min * effective_upscale_factor_x)
            x_max = x_min + target_x_size
            y_min = round(y_min * effective_upscale_factor_y)
            y_max = y_min + target_y_size
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1
        # Ensure width and height are within specified bounds, key for ranged and forced size
        if mode == 'ranged size' or mode == 'forced size':
            if x_size < min_width:
                x_max = min(x_max + (min_width - x_size), width - 1)
            elif x_size > max_width:
                x_max = x_min + max_width - 1
            if y_size < min_height:
                y_max = min(y_max + (min_height - y_size), height - 1)
            elif y_size > max_height:
                y_max = y_min + max_height - 1
        # Recalculate x_size and y_size after adjustments
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1
        # Pad area (if possible, i.e. if pad is smaller than width/height) to avoid the sampler returning smaller results
        if (mode == 'free size' or mode == 'ranged size') and padding > 1:
            x_min, x_max = self.apply_padding(x_min, x_max, width, padding)
            y_min, y_max = self.apply_padding(y_min, y_max, height, padding)
        # Ensure that context area doesn't go outside of the image
        x_min = max(x_min, 0)
        x_max = min(x_max, width - 1)
        y_min = max(y_min, 0)
        y_max = min(y_max, height - 1)
        # Crop the image and the mask, sized context area
        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[:, y_min:y_max+1, x_min:x_max+1]
        cropped_mask_blend = blend_mask[:, y_min:y_max+1, x_min:x_max+1]
        # Grow and blur mask for blend if requested
        if blend_pixels > 0.001:
            cropped_mask_blend = self.grow_and_blur_mask(cropped_mask_blend, blend_pixels)
        # Return stitch (to be consumed by the class below), image, and mask
        stitch = {'x': x_min, 'y': y_min, 'original_image': original_image, 'cropped_mask_blend': cropped_mask_blend, 'rescale_x': effective_upscale_factor_x, 'rescale_y': effective_upscale_factor_y, 'start_x': start_x, 'start_y': start_y, 'initial_width': initial_width, 'initial_height': initial_height}
        return (stitch, cropped_image, cropped_mask)
class InpaintStitch:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch
    This node stitches the inpainted image without altering unmasked areas.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch": ("STITCH",),
                "inpainted_image": ("IMAGE",),
                "rescale_algorithm": (["nearest", "bilinear", "bicubic", "bislerp", "lanczos", "box", "hamming"], {"default": "bislerp"}),
            }
        }
    CATEGORY = "CCNotes/Third-party/comfyui-inpaint-cropandstitch"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint_stitch"
    # This function is from comfy_extras: https://github.com/comfyanonymous/ComfyUI
    def composite(self, destination, source, x, y, mask=None, multiplier=8, resize_source=False):
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(destination.shape[2], destination.shape[3]), mode="bilinear")
        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])
        x = max(-source.shape[3] * multiplier, min(x, destination.shape[3] * multiplier))
        y = max(-source.shape[2] * multiplier, min(y, destination.shape[2] * multiplier))
        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + source.shape[3], top + source.shape[2],)
        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])
        # calculate the bounds of the source that will be overlapping the destination
        # this prevents the source trying to overwrite latent pixels that are out of bounds
        # of the destination
        visible_width, visible_height = (destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)
        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask
        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask  * destination[:, :, top:bottom, left:right]
        destination[:, :, top:bottom, left:right] = source_portion + destination_portion
        return destination
    def inpaint_stitch(self, stitch, inpainted_image, rescale_algorithm):
        results = []
        batch_size = inpainted_image.shape[0]
        assert len(stitch['x']) == batch_size, "Stitch size doesn't match image batch size"
        for b in range(batch_size):
            one_image = inpainted_image[b]
            one_stitch = {}
            for key in stitch:
                # Extract the value at the specified index and assign it to the single_stitch dictionary
                one_stitch[key] = stitch[key][b]
            one_image = one_image.unsqueeze(0)
            one_image, = self.inpaint_stitch_single_image(one_stitch, one_image, rescale_algorithm)
            one_image = one_image.squeeze(0)
            results.append(one_image)
        # Stack the results to form a batch
        result_batch = torch.stack(results, dim=0)
        return (result_batch,)
    def inpaint_stitch_single_image(self, stitch, inpainted_image, rescale_algorithm):
        original_image = stitch['original_image']
        cropped_mask_blend = stitch['cropped_mask_blend']
        x = stitch['x']
        y = stitch['y']
        stitched_image = original_image.clone().movedim(-1, 1)
        start_x = stitch['start_x']
        start_y = stitch['start_y']
        initial_width = stitch['initial_width']
        initial_height = stitch['initial_height']
        inpaint_width = inpainted_image.shape[2]
        inpaint_height = inpainted_image.shape[1]
        if stitch['rescale_x'] < 0.999 or stitch['rescale_x'] > 1.001 or stitch['rescale_y'] < 0.999 or stitch['rescale_y'] > 1.001:
            samples = inpainted_image.movedim(-1, 1)
            width = math.ceil(float(inpaint_width)/stitch['rescale_x'])+1
            height = math.ceil(float(inpaint_height)/stitch['rescale_y'])+1
            x = math.floor(float(x)/stitch['rescale_x'])
            y = math.floor(float(y)/stitch['rescale_y'])
            samples = rescale(samples, width, height, rescale_algorithm)
            inpainted_image = samples.movedim(1, -1)
            
            samples = cropped_mask_blend.movedim(-1, 1)
            samples = samples.unsqueeze(0)
            samples = rescale(samples, width, height, rescale_algorithm)
            samples = samples.squeeze(0)
            cropped_mask_blend = samples.movedim(1, -1)
            cropped_mask_blend = torch.clamp(cropped_mask_blend, 0.0, 1.0)
        output = self.composite(stitched_image, inpainted_image.movedim(-1, 1), x, y, cropped_mask_blend, 1).movedim(1, -1)
        # Crop out from the extended dimensions back to original.
        cropped_output = output[:, start_y:start_y + initial_height, start_x:start_x + initial_width, :]
        output = cropped_output
        return (output,)
class InpaintExtendOutpaint:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch
    This node extends an image for inpainting with Inpaint Crop and Stitch.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": (["factors", "pixels"], {"default": "factors"}),
                "expand_up_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "expand_up_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01}),
                "expand_down_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "expand_down_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01}),
                "expand_left_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "expand_left_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01}),
                "expand_right_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "expand_right_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "optional_context_mask": ("MASK",),
            }
        }
    CATEGORY = "CCNotes/Third-party/comfyui-inpaint-cropandstitch"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "context_mask")
    FUNCTION = "inpaint_extend"
    def inpaint_extend(self, image, mask, mode, expand_up_pixels, expand_up_factor, expand_down_pixels, expand_down_factor, expand_left_pixels, expand_left_factor, expand_right_pixels, expand_right_factor, optional_context_mask=None):
        assert image.shape[0] == mask.shape[0], "Batch size of images and masks must be the same"
        if optional_context_mask is not None:
            assert optional_context_mask.shape[0] == image.shape[0], "Batch size of optional_context_masks must be the same as images or None"
        results_image = []
        results_mask = []
        results_context_mask = []
        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)  # Adding batch dimension
            one_mask = mask[b].unsqueeze(0)    # Adding batch dimension
            one_context_mask = None
            if optional_context_mask is not None:
                one_context_mask = optional_context_mask[b].unsqueeze(0)
            #Validate or initialize mask
            if one_mask.shape[1] != one_image.shape[1] or one_mask.shape[2] != one_image.shape[2]:
                non_zero_indices = torch.nonzero(one_mask[0], as_tuple=True)
                if not non_zero_indices[0].size(0):
                    one_mask = torch.zeros_like(one_image[:, :, :, 0])
                else:
                    assert False, "mask size must match image size"
            # Validate or initialize context mask
            if one_context_mask is not None and (one_context_mask.shape[1] != one_image.shape[1] or one_context_mask.shape[2] != one_image.shape[2]):
                non_zero_indices = torch.nonzero(one_context_mask[0], as_tuple=True)
                if not non_zero_indices[0].size(0):
                    one_context_mask = torch.zeros_like(one_image[:, :, :, 0])
                else:
                    assert False, "context_mask size must match image size"
            # Get original dimensions
            orig_height, orig_width = one_image.shape[1], one_image.shape[2]
            if mode == "factors":
                # Calculate new dimensions based on factors
                new_height = int(orig_height * (expand_up_factor + expand_down_factor - 1))
                new_width = int(orig_width * (expand_left_factor + expand_right_factor - 1))
                up_padding = int(orig_height * (expand_up_factor - 1))
                down_padding = new_height - orig_height - up_padding
                left_padding = int(orig_width * (expand_left_factor - 1))
                right_padding = new_width - orig_width - left_padding
            elif mode == "pixels":
                # Calculate new dimensions based on pixel expansion
                new_height = orig_height + expand_up_pixels + expand_down_pixels
                new_width = orig_width + expand_left_pixels + expand_right_pixels
                up_padding = expand_up_pixels
                down_padding = expand_down_pixels
                left_padding = expand_left_pixels
                right_padding = expand_right_pixels
            # Expand image
            new_image = torch.zeros((one_image.shape[0], new_height, new_width, one_image.shape[3]), dtype=one_image.dtype)
            new_image[:, up_padding:up_padding + orig_height, left_padding:left_padding + orig_width, :] = one_image.squeeze(0)
            start_y = up_padding
            start_x = left_padding
            initial_height = orig_height
            initial_width = orig_width
            # Mirror image so there's no bleeding of black border when using inpaintmodelconditioning
            available_top = min(start_y, initial_height)
            available_bottom = min(new_height - (start_y + initial_height), initial_height)
            available_left = min(start_x, initial_width)
            available_right = min(new_width - (start_x + initial_width), initial_width)
            # Top
            if available_top:
                new_image[:, start_y - available_top:start_y, start_x:start_x + initial_width, :] = torch.flip(image[:, :available_top, :, :], [1])
            # Bottom
            if available_bottom:
                new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x:start_x + initial_width, :] = torch.flip(image[:, -available_bottom:, :, :], [1])
            # Left
            if available_left:
                new_image[:, start_y:start_y + initial_height, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x:start_x + available_left, :], [2])
            # Right
            if available_right:
                new_image[:, start_y:start_y + initial_height, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [2])
            # Top-left corner
            if available_top and available_left:
                new_image[:, start_y - available_top:start_y, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x:start_x + available_left, :], [1, 2])
            # Top-right corner
            if available_top and available_right:
                new_image[:, start_y - available_top:start_y, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y:start_y + available_top, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])
            # Bottom-left corner
            if available_bottom and available_left:
                new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x - available_left:start_x, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x:start_x + available_left, :], [1, 2])
            # Bottom-right corner
            if available_bottom and available_right:
                new_image[:, start_y + initial_height:start_y + initial_height + available_bottom, start_x + initial_width:start_x + initial_width + available_right, :] = torch.flip(new_image[:, start_y + initial_height - available_bottom:start_y + initial_height, start_x + initial_width - available_right:start_x + initial_width, :], [1, 2])
            # Expand mask
            new_mask = torch.ones((one_mask.shape[0], new_height, new_width), dtype=one_mask.dtype)
            new_mask[:, up_padding:up_padding + orig_height, left_padding:left_padding + orig_width] = one_mask.squeeze(0)
            # Expand context mask if present
            if one_context_mask is not None:
                new_context_mask = torch.zeros((one_context_mask.shape[0], new_height, new_width), dtype=one_context_mask.dtype)
                new_context_mask[:, up_padding:up_padding + orig_height, left_padding:left_padding + orig_width] = one_context_mask.squeeze(0)
            # Append results
            results_image.append(new_image.squeeze(0))
            results_mask.append(new_mask.squeeze(0))
            if one_context_mask is not None:
                results_context_mask.append(new_context_mask.squeeze(0))
        # Stack the results to form batches
        output_image = torch.stack(results_image, dim=0)
        output_mask = torch.stack(results_mask, dim=0)
        output_context_mask = None
        if optional_context_mask is not None:
            output_context_mask = torch.stack(results_context_mask, dim=0)
        return (output_image, output_mask, output_context_mask)
class InpaintResize:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch
    This node resizes an image before inpainting with Inpaint Crop and Stitch.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "rescale_algorithm": (["nearest", "bilinear", "bicubic", "bislerp", "lanczos", "box", "hamming"], {"default": "bicubic"}),
                "mode": (["ensure minimum size", "factor"], {"default": "ensure minimum size"}),
                "min_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}), # ranged
                "min_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}), # ranged
                "rescale_factor": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 100.0, "step": 0.01}), # free
            },
            "optional": {
                "optional_context_mask": ("MASK",),
            }
        }
    CATEGORY = "CCNotes/Third-party/comfyui-inpaint-cropandstitch"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "context_mask")
    FUNCTION = "inpaint_resize"
    def inpaint_resize(self, image, mask, rescale_algorithm, mode, min_width, min_height, rescale_factor, optional_context_mask=None):
        assert image.shape[0] == mask.shape[0], "Batch size of images and masks must be the same"
        if optional_context_mask is not None:
            assert optional_context_mask.shape[0] == image.shape[0], "Batch size of optional_context_masks must be the same as images or None"
        results_image = []
        results_mask = []
        results_context_mask = []
        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)  # Adding batch dimension
            one_mask = mask[b].unsqueeze(0)    # Adding batch dimension
            one_context_mask = None
            if optional_context_mask is not None:
                one_context_mask = optional_context_mask[b].unsqueeze(0)
            #Validate or initialize mask
            if one_mask.shape[1] != one_image.shape[1] or one_mask.shape[2] != one_image.shape[2]:
                non_zero_indices = torch.nonzero(one_mask[0], as_tuple=True)
                if not non_zero_indices[0].size(0):
                    one_mask = torch.zeros_like(one_image[:, :, :, 0])
                else:
                    assert False, "mask size must match image size"
            # Validate or initialize context mask
            if one_context_mask is not None and (one_context_mask.shape[1] != one_image.shape[1] or one_context_mask.shape[2] != one_image.shape[2]):
                non_zero_indices = torch.nonzero(one_context_mask[0], as_tuple=True)
                if not non_zero_indices[0].size(0):
                    one_context_mask = torch.zeros_like(one_image[:, :, :, 0])
                else:
                    assert False, "context_mask size must match image size"
            # Get original dimensions
            orig_height, orig_width = one_image.shape[1], one_image.shape[2]
            # Calculate target width and height
            if mode == "ensure minimum size":
                # Start with original dimensions
                width = orig_width
                height = orig_height
                # If either dimension is smaller than the minimum, scale up
                if orig_width < min_width or orig_height < min_height:
                    aspect_ratio = orig_width / orig_height
                    if min_width / aspect_ratio >= min_height:
                        width = min_width
                        height = int(min_width / aspect_ratio)
                    else:
                        height = min_height
                        width = int(min_height * aspect_ratio)
                # Ensure the dimensions are at least min_width and min_height
                width = max(width, min_width)
                height = max(height, min_height)
            elif mode == "factor":
                width = round(orig_width * rescale_factor)
                height = round(orig_height * rescale_factor)
            # Resize
            if orig_width != width or orig_height != height:
                samples = one_image            
                samples = samples.movedim(-1, 1)
                samples = rescale(samples, width, height, rescale_algorithm)
                samples = samples.movedim(1, -1)
                one_image = samples
                samples = one_mask
                samples = samples.unsqueeze(1)
                samples = rescale(samples, width, height, "nearest")
                samples = samples.squeeze(1)
                one_mask = samples
                if one_context_mask is not None:
                    samples = one_context_mask
                    samples = samples.unsqueeze(1)
                    samples = rescale(samples, width, height, "nearest")
                    samples = samples.squeeze(1)
                    one_context_mask = samples
            # Append results
            results_image.append(one_image.squeeze(0))
            results_mask.append(one_mask.squeeze(0))
            if one_context_mask is not None:
                results_context_mask.append(one_context_mask.squeeze(0))
        # Stack the results to form batches
        output_image = torch.stack(results_image, dim=0)
        output_mask = torch.stack(results_mask, dim=0)
        output_context_mask = None
        if optional_context_mask is not None:
            output_context_mask = torch.stack(results_context_mask, dim=0)
        return (output_image, output_mask, output_context_mask)
class InpaintCropImproved:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic"}),
                "preresize": ("BOOLEAN", {"default": False, "tooltip": "Resize the original image before processing."}),
                "preresize_mode": (["ensure minimum resolution", "ensure maximum resolution", "ensure minimum and maximum resolution"], {"default": "ensure minimum resolution"}),
                "preresize_min_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_width": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_height": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "mask_fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Mark as masked any areas fully enclosed by mask."}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Expand the mask by a certain amount of pixels before processing."}),
                "mask_invert": ("BOOLEAN", {"default": False,"tooltip": "Invert mask so that anything masked will be kept."}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1, "tooltip": "How many pixels to blend into the original image."}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Ignore mask values lower than this value."}),
                "extend_for_outpainting": ("BOOLEAN", {"default": False, "tooltip": "Extend the image for outpainting."}),
                "extend_up_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_down_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_left_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01, "tooltip": "Grow the context area from the mask by a certain factor in every direction. For example, 1.5 grabs extra 50% up, down, left, and right as context."}),
                "output_resize_to_target_size": ("BOOLEAN", {"default": True, "tooltip": "Force a specific resolution for sampling."}),
                "output_target_width": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "32"}),
           },
           "optional": {
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),
           }
        }
    FUNCTION = "inpaint_crop"
    CATEGORY = "CCNotes/Third-party/comfyui-inpaint-cropandstitch"
    DESCRIPTION = "Crops an image around a mask for inpainting, the optional context mask defines an extra area to keep for the context."
    DEBUG_MODE = False
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")
    def inpaint_crop(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, mask=None, optional_context_mask=None):
        image = image.clone()
        if mask is not None:
            mask = mask.clone()
        if optional_context_mask is not None:
            optional_context_mask = optional_context_mask.clone()
        output_padding = int(output_padding)
        # Check that some parameters make sense
        if preresize and preresize_mode == "ensure minimum and maximum resolution":
            assert preresize_max_width >= preresize_min_width, "Preresize maximum width must be greater than or equal to minimum width"
            assert preresize_max_height >= preresize_min_height, "Preresize maximum height must be greater than or equal to minimum height"
        if image.shape[0] > 1:
            assert output_resize_to_target_size, "output_resize_to_target_size must be enabled when input is a batch of images, given all images in the batch output have to be the same size"
        # When a LoadImage node passes a mask without user editing, it may be the wrong shape.
        # Detect and fix that to avoid shape mismatch errors.
        if mask is not None and (image.shape[0] == 1 or mask.shape[0] == 1 or mask.shape[0] == image.shape[0]):
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(mask) == 0:
                    mask = torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)
        if optional_context_mask is not None and (image.shape[0] == 1 or optional_context_mask.shape[0] == 1 or optional_context_mask.shape[0] == image.shape[0]):
            if optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(optional_context_mask) == 0:
                    optional_context_mask = torch.zeros((optional_context_mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)
        # If no mask is provided, create one with the shape of the image
        if mask is None:
            mask = torch.zeros_like(image[:, :, :, 0])
        # If there is only one image for many masks, replicate it for all masks
        if mask.shape[0] > 1 and image.shape[0] == 1:
            assert image.dim() == 4, f"Expected 4D BHWC image tensor, got {image.shape}"
            image = image.expand(mask.shape[0], -1, -1, -1).clone()
        # If there is only one mask for many images, replicate it for all images
        if image.shape[0] > 1 and mask.shape[0] == 1:
            assert mask.dim() == 3, f"Expected 3D BHW mask tensor, got {mask.shape}"
            mask = mask.expand(image.shape[0], -1, -1).clone()
        # If no optional_context_mask is provided, create one with the shape of the image
        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(image[:, :, :, 0])
        # If there is only one optional_context_mask for many images, replicate it for all images
        if image.shape[0] > 1 and optional_context_mask.shape[0] == 1:
            assert optional_context_mask.dim() == 3, f"Expected 3D BHW optional_context_mask tensor, got {optional_context_mask.shape}"
            optional_context_mask = optional_context_mask.expand(image.shape[0], -1, -1).clone()
         # Validate data
        assert image.ndimension() == 4, f"Expected 4 dimensions for image, got {image.ndimension()}"
        assert mask.ndimension() == 3, f"Expected 3 dimensions for mask, got {mask.ndimension()}"
        assert optional_context_mask.ndimension() == 3, f"Expected 3 dimensions for optional_context_mask, got {optional_context_mask.ndimension()}"
        assert mask.shape[1:] == image.shape[1:3], f"Mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {mask.shape[1:]}"
        assert optional_context_mask.shape[1:] == image.shape[1:3], f"optional_context_mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {optional_context_mask.shape[1:]}"
        assert mask.shape[0] == image.shape[0], f"Mask batch does not match image batch. Expected {image.shape[0]}, got {mask.shape[0]}"
        assert optional_context_mask.shape[0] == image.shape[0], f"Optional context mask batch does not match image batch. Expected {image.shape[0]}, got {optional_context_mask.shape[0]}"
        # Run for each image separately
        result_stitcher = {
            'downscale_algorithm': downscale_algorithm,
            'upscale_algorithm': upscale_algorithm,
            'blend_pixels': mask_blend_pixels,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
        }
        result_image = []
        result_mask = []
        batch_size = image.shape[0]
        for b in range(batch_size):
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = optional_context_mask[b].unsqueeze(0)
            outputs = self.inpaint_crop_single_image(
                one_image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode,
                preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height,
                extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor,
                mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels,
                context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height,
                output_padding, one_mask, one_optional_context_mask)
            stitcher, cropped_image, cropped_mask = outputs[:3]
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                result_stitcher[key].append(stitcher[key])
            cropped_image = cropped_image.clone().squeeze(0)
            result_image.append(cropped_image)
            cropped_mask = cropped_mask.clone().squeeze(0)
            result_mask.append(cropped_mask)
        result_image = torch.stack(result_image, dim=0)
        result_mask = torch.stack(result_mask, dim=0)
        return result_stitcher, result_image, result_mask
    def inpaint_crop_single_image(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, mask, optional_context_mask):
        if preresize:
            image, mask, optional_context_mask = preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height)
        if mask_fill_holes:
           mask = fillholes_iterative_hipass_fill_m(mask)
        if mask_expand_pixels > 0:
            mask = expand_m(mask, mask_expand_pixels)
        if mask_invert:
            mask = invert_m(mask)
        if mask_blend_pixels > 0:
            mask = expand_m(mask, mask_blend_pixels)
            mask = blur_m(mask, mask_blend_pixels*0.5)
        if mask_hipass_filter >= 0.01:
            mask = hipassfilter_m(mask, mask_hipass_filter)
            optional_context_mask = hipassfilter_m(optional_context_mask, mask_hipass_filter)
        if extend_for_outpainting:
            image, mask, optional_context_mask = extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor)
        context, x, y, w, h = findcontextarea_m(mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if context_from_mask_extend_factor >= 1.01:
            context, x, y, w, h = growcontextarea_m(context, mask, x, y, w, h, context_from_mask_extend_factor)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        context, x, y, w, h = combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if not output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, w, h, output_padding, downscale_algorithm, upscale_algorithm)
        else: # if output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, output_target_width, output_target_height, output_padding, downscale_algorithm, upscale_algorithm)
        # For blending, grow the mask even further and make it blurrier.
        cropped_mask_blend = cropped_mask.clone()
        if mask_blend_pixels > 0:
           cropped_mask_blend = blur_m(cropped_mask_blend, mask_blend_pixels*0.5)
        stitcher = {
            'canvas_to_orig_x': cto_x,
            'canvas_to_orig_y': cto_y,
            'canvas_to_orig_w': cto_w,
            'canvas_to_orig_h': cto_h,
            'canvas_image': canvas_image,
            'cropped_to_canvas_x': ctc_x,
            'cropped_to_canvas_y': ctc_y,
            'cropped_to_canvas_w': ctc_w,
            'cropped_to_canvas_h': ctc_h,
            'cropped_mask_for_blend': cropped_mask_blend,
        }
        return stitcher, cropped_image, cropped_mask
class InpaintStitchImproved:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch
    This node stitches the inpainted image without altering unmasked areas.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
            }
        }
    CATEGORY = "CCNotes/Third-party/comfyui-inpaint-cropandstitch"
    DESCRIPTION = "Stitches an image cropped with Inpaint Crop back into the original image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint_stitch"
    def inpaint_stitch(self, stitcher, inpainted_image):
        inpainted_image = inpainted_image.clone()
        results = []
        batch_size = inpainted_image.shape[0]
        assert len(stitcher['cropped_to_canvas_x']) == batch_size or len(stitcher['cropped_to_canvas_x']) == 1, "Stitch batch size doesn't match image batch size"
        override = False
        if len(stitcher['cropped_to_canvas_x']) != batch_size and len(stitcher['cropped_to_canvas_x']) == 1:
            override = True
        for b in range(batch_size):
            one_image = inpainted_image[b]
            one_stitcher = {}
            for key in ['downscale_algorithm', 'upscale_algorithm', 'blend_pixels']:
                one_stitcher[key] = stitcher[key]
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                if override: # One stitcher for many images, always read 0.
                    one_stitcher[key] = stitcher[key][0]
                else:
                    one_stitcher[key] = stitcher[key][b]
            one_image = one_image.unsqueeze(0)
            one_image, = self.inpaint_stitch_single_image(one_stitcher, one_image)
            one_image = one_image.squeeze(0)
            one_image = one_image.clone()
            results.append(one_image)
        result_batch = torch.stack(results, dim=0)
        return (result_batch,)
    def inpaint_stitch_single_image(self, stitcher, inpainted_image):
        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm = stitcher['upscale_algorithm']
        canvas_image = stitcher['canvas_image']
        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']
        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']
        cto_w = stitcher['canvas_to_orig_w']
        cto_h = stitcher['canvas_to_orig_h']
        mask = stitcher['cropped_mask_for_blend']  # shape: [1, H, W]
        output_image = stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)
        return (output_image,)
