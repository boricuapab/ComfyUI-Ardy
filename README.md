# ComfyUI-Ardy

Run ARDY locally with an animated body mesh and ground-click motion steering.

## Live preview

Add **ARDY Live Preview** and connect **Load ARDY Model**. Click
**Start new take**, then click the ground grid to append destinations. Drag to orbit
and scroll to zoom; right-drag pans without opening the graph menu. All clicked
waypoints stay marked until **Clear route**, or edit `waypoints` as `[[x,z], [x,z], ...]`. The character follows
them in order. Click inside the viewport and hold arrow keys for camera-relative
steering; releasing the keys resumes the waypoint route. Keyboard control ends
when the viewport loses focus. Change the prompt and speed while running; Pause, Resume,
Stop & save take, Clear route and camera-follow controls are in the preview toolbar.
Choose Mesh, Mesh + skeleton, or Skeleton without restarting generation.

Steering conditions the next generated motion chunk. The Core checkpoint generates
40 frames at 20 FPS, so direction changes can take roughly two seconds to appear.
`run_seconds` defaults to 120; set it to 0 to run until stopped. Closing the viewer
stops its session after a short disconnect timeout. Live generation occupies the
ComfyUI queue until it stops. The body uses ARDY's bundled Core character skin.

## Record and save

Connect **ARDY Live Preview** to **ARDY Motion to GLB** and a native 3D preview/save node.
**Start new take** records only the live node. **Stop & save take** finishes it.
Run the workflow to execute connected GLB and video nodes, or use the camera
node's **Load / render frames** button to render only that branch.
GLB files are saved under `ComfyUI/output/ardy/recording_*.glb`.

Completed takes are cached by ComfyUI and saved under `output/ardy/recordings`.
Ordinary runs reuse the take, including after restart or cache eviction, without
loading ARDY. **Start new take** creates a new recording with the current settings.
Saved takes remain until you remove their files. ARDY and the text encoder use
ComfyUI model patchers and offload after execution; downstream MiniMax work does
not retain their model weights in VRAM.

The live node's `recorded_motion` output includes the animated body and skeleton.
Recording follows playback time: pauses and generated-but-unplayed frames are
excluded. The duration limit measures played motion, excluding model loading.
Use Stop & save take rather than queue cancel to preserve the take. Mesh GLB size grows
with recording duration.

### Driving video frames

Connect recorded motion to **ARDY Preview Advanced · Driving Frames**. Its IMAGE
output is an ordered frame batch for video/image workflows; MASK is the body
silhouette and fps is the playback rate. The node shows an animated rendered preview.
Use solid, normal, or depth rendering, set resolution and camera angles, or connect
`camera_info` from a native 3D camera node. `frame_step` skips samples and adjusts
fps to retain playback speed. Rendering happens after the live recording stops.
The GLB branch retains the interactive Preview 3D (Advanced) viewer.

### Camera recording

Connect `recorded_motion` to **ARDY Camera Recorder**, then click **Load / render
frames**. Orbit with left-drag, pan with right-drag, and zoom with the wheel.
**Use current camera** saves static framing. **Record camera** plays from the
beginning while recording your movements; **Stop camera** saves the path. The
camera holds its last position if you stop early. Click **Load / render frames**
again to apply the path. Camera paths are saved in the workflow.

Camera Recorder remembers the completed take independently of its live input.
Use **Load NPZ…** to browse for a previously recorded ARDY take, or **Use connected
motion** to switch back to the input. Camera-only renders reuse that saved take
without loading ARDY. Wheel scrolling zooms inside the camera viewport.

**Save camera JSON** exports the recorded path. **Load camera JSON…** restores it.
To share a path between render nodes, add **Load ARDY Camera**, browse for the JSON,
and connect its `camera_info` output. ARDY render nodes play the full timed path;
native static-camera consumers use its first pose. A connected path takes priority
over the recorder's local path.

Connect its `images` and `fps` to native **Create Video**, or its `images` to a
reference-video frame input. `output_fps` defaults to 24 for MiniMax H3 and
resamples ARDY's 20 fps motion without changing duration. The grid is an editing
guide and is excluded from rendered frames. The viewport uses the output aspect
ratio; final lighting comes from ComfyUI's mesh renderer.

## Workflow

Build a generation workflow with:

`Load ARDY Text Encoder` → `Load ARDY Model` → `Generate ARDY Motion` → `ARDY Motion to GLB` → `Preview 3D (Advanced)`

Connect the text encoder loader to the model loader's `text_encoder` socket.

Use the viewer's animation controls to play or scrub the motion. Select mesh,
mesh and skeleton, or skeleton in `ARDY Motion to GLB`. Core body animation is
exported as mesh morph targets; other skeletons retain the bone preview. It also connects to ComfyUI's
3D file saving nodes. `fps` controls playback speed; generation uses the checkpoint's
native frame rate. Denoising steps are capped to the checkpoint's schedule (10 for Core Horizon40).

## Local files

Install `requirements.txt` in the Python environment that runs ComfyUI, then restart it.
The requirements install a pinned revision of [NVIDIA ARDY](https://github.com/nv-tlabs/ardy)
into the same Python environment. Git and a C++17 build toolchain are required by
upstream ARDY (on Windows, Visual Studio Build Tools with Desktop development with
C++). Its build uses CMake and may download pybind11 and Eigen.
The optional local `vendor/` directory is excluded from this repository.

Place the complete motion checkpoint, including its `stats` directory, in
`ComfyUI/models/ardy/ARDY-Core-RP-20FPS-Horizon40`.

The text encoder uses these folders under `ComfyUI/models/text_encoders/ardy`:

- `Meta-Llama-3-8B-Instruct`: base model safetensors, index and config.
- `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`: MNTP adapter, config and tokenizer files.
- `LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`: supervised adapter and config.

Weights must be downloaded separately. Node execution uses local files only and
never downloads models. `Load ARDY Text Encoder` exposes model selection and
`device` independently of the motion model's `device`. Text encoding
defaults to `cuda:0` (GPU 0); choose `cpu` to leave VRAM for motion generation.
The original BF16 encoder needs about 16 GB for its weights; INT8 uses less memory.
CPU prompt encoding is slower. ComfyUI owns the lifetime of loaded node outputs; there is
no separate global model cache.

### INT8 ConvRot text encoder

`Load ARDY Text Encoder` has a `text_encoder` selector. `auto` prefers
`text_encoders/ardy/Meta-Llama-3-8B-Instruct_Int8_Convrot/meta_llama_3_8B_Instruct_comfy_int8_convrot.safetensors`
automatically, downloading it when missing. Select `int8_convrot` to require it, or `bf16` for the original
Hugging Face shards (those shards must still be installed).

The INT8 file contains the base Llama weights only. Both local LLM2Vec adapter
folders above remain required. ComfyUI applies their LoRA contributions during
encoding without dequantizing and merging the entire base model. The encoder
uses ComfyUI quantized operations, bidirectional attention, and model offloading.
No conversion is needed.

The recording workflow has been verified with dynamic VRAM enabled.

Verification: `python -m unittest discover -s tests -v`.

## Installation

From your ComfyUI `custom_nodes` directory:

```sh
git clone https://github.com/boricuapab/ComfyUI-Ardy.git
python -m pip install -r ComfyUI-Ardy/requirements.txt
```

Use the Python interpreter that runs ComfyUI, then restart ComfyUI.
Model weights and recordings are not included.

## Licenses

The wrapper is MIT licensed. ARDY is installed separately under its upstream
license. The bundled Three.js files retain their MIT license in
`web/vendor/LICENSE.three`.

### Automatic model downloads

Models are stored under the configured ComfyUI models directory in ardy/. The default Core Horizon40 checkpoint includes its statistics. Text encoders are stored in text_encoders/ardy/. The auto option prefers the ungated INT8 ConvRot checkpoint from boricuapab/meta_llama_3_8B_Instruct_comfy_int8_convrot and downloads both McGill-NLP LLM2Vec adapters. Selecting bf16 downloads the official Meta-Llama-3-8B-Instruct base instead; access to that gated repository requires approval and Hugging Face authentication. Existing complete files are reused. hf-xet enables Xet transfer support where available. Downloads happen only when a loader executes, never during node discovery.
