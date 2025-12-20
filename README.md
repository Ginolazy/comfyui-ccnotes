# CCNotes Custom Nodes for ComfyUI 🎨⚡

A fun and powerful collection of utility, image processing, and text manipulation nodes for ComfyUI. Designed to make your workflow smoother, more creative, and a bit magical. ✨

## Features

- **Image & Mask Processing**: Blend, constrain, switch, swap, and scale images and masks like a pro.  
- **Process & Restore**: Crop images by mask, process them, and seamlessly restore them to their original positions—perfect for inpainting adventures.  
- **Text Manipulation**: Crunch numbers, concatenate strings, translate text, and display content in your workflow.  
- **Inpainting Utilities**: Specialized nodes for cropping, stitching, and extending images effortlessly.  
- **Workflow Control**: Logical switches, pauses, previews, and automatic muting to keep your workflow under control (or secretly do the work for you). 😏

## Node Highlights
### 🔀 SwitchAnyPro

More than just a basic switch, **SwitchAnyPro** also **controls execution flow**.  

**Key feature:**  
- 💤 **Upstream silence**  
  Unselected inputs keep their entire upstream node chains completely inactive. Only the selected branch runs, saving computation and avoiding side effects.  

**Why it’s awesome:**  
- Cuts unnecessary processing, especially for heavy image/mask chains  
- Prevents side effects from inactive branches  
- Makes complex workflows clear, deterministic, and easy to debug  
- Enables true conditional execution, just like a mini-program inside ComfyUI  

**Typical use cases:**  
- Prompt / conditioning A–B testing  
- Switching between image and mask processing branches  
- Debugging workflows without deleting nodes  
- Performance-friendly pipelines  

**One-line summary:**  
> **SwitchAnyPro**: Only the selected branch executes, while all other upstream branches stay silently in the background 😎  

---

### 🔇 AutoMute

Automatically mutes or unmutes target groups or nodes based on monitored node states.  

**Key feature:**  
- 🤖 **Smart auto-control**  
  Groups or nodes only wake up if a monitored node is active. Otherwise, they stay silent, keeping your workflow neat and efficient.  

**Why it’s awesome:**  
- Saves computation and avoids unnecessary processing  
- Works seamlessly with Fast Groups Muter  
- Lets you set up reactive workflows without extra wiring  

**Typical use cases:**  
- Automatically enable mask-processing groups only when needed  
- Keep preview or auxiliary nodes dormant until triggered  
- Dynamically control groups based on workflow logic  

**One-line summary:**  
> **AutoMute**: Your workflow’s smart ninja—activates only when needed, stays silent when not 😎  

---

### 👀 AnyPreviewPause

A multi-purpose node for previews and workflow intervention.  

**Key features:**  
- 🖼️ **Image & Mask pairing** – Automatically pairs connected images and masks for grouped overlay previews, supports lists  
- ⏸️ **Pause execution** – Pause before continuing to let you tweak masks or text  
- ✨ **Flexible previews** – Works with images, masks, and text  

**Why it’s awesome:**  
- Lets you peek at intermediate results without breaking the workflow  
- Perfect for interactive inpainting or text editing  
- Makes debugging and fine-tuning fun and safe  

**One-line summary:**  
> **AnyPreviewPause**: Peek, pause, and tweak—preview anything while keeping the workflow under control 😎  

---

## Nodes
### Image & Mask
* `BlendByMask`: Blend two images using a mask—smooth magic.  
* `ImageMask_Constrain`: Keep your images and masks perfectly sized.  
* `ImageMask_Switch`, `ImageMask_SwitchAuto`: Swap images or masks on the fly.  
* `ImageMask_Swap`: Flip two images/masks like a card trick.  
* `ImageMask_Transform`: Rotate, flip, and play with images/masks.  
* `ImageBatchToImageList`, `ImageListToImageBatch`: Convert formats effortlessly.  
* `MakeBatch`: Turn multiple images into a batch with style.  
* `ScaleAny`, `ImageMask_Scale`: Resize images or masks.  
* `SwitchMaskAuto`: Let masks switch themselves.  
* `ImageBlank`: Make a blank canvas—pure creative freedom.  
* `ImageFilterAdjustments`: Brightness, contrast, and mood tweaks.  
* `ImageRemoveAlpha`: Say goodbye to alpha channels.  
* `ImageSwap`: Swap images with a snap.  
* `ImageMaskComposite`: Layer masks over images effortlessly.

### Process & Restore
* `CropByMask`, `CropByMaskRestore`: Crop and restore like a workflow ninja.  
* `ImageConcat`, `ImageConcatRestore`: Stitch images together and bring them back.  
* `ImageMask_Scale`, `ImageMask_ScaleRestore`: Resize, then restore—no sweat.

### Text
* `Float`, `Int`: Basic numeric nodes.  
* `MathOperationFloat`, `MathOperationInt`: Crunch numbers with ease.  
* `StringListToString`: Join string lists into one.  
* `TextConcat`: Concatenate texts effortlessly.  
* `ShowText`: Display content in your workflow.  
* `TextMultiline`: 🤫 Shh… works with macOS Shortcuts to translate and transform text magically. Fully customizable, Mac-only fun!

### Utilities & Logic
* `AnyPreview`: Preview images/text, optionally pause execution.  
* `AnyPause`: Hit pause whenever you need.  
* `AnyPreviewPause`: Auto-pairs connected images and masks for overlay previews and can pause the workflow—perfect for fine-tuning. 😎  
* `SwitchAny`, `SwitchAnyBasic`, `SwitchAuto`, `SwitchOutput`: Logic switches for anything you throw at them.  
* `PrimitiveAdvanced`: Advanced primitive node with flair.  
* `MakeAnyList`: Create lists of any type.  
* `AutoMute`: Automatically mutes/unmutes target groups/nodes based on monitored states—works like a secret autopilot with Fast Groups Muter.

## Installation

1. Navigate to your ComfyUI `custom_nodes` directory.  
2. Clone this repository:  
	```bash
	git clone https://github.com/Ginolazy/ComfyUI-CCNotes.git
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
 
