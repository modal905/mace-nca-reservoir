"""
Generate side-by-side CA dynamics animation (baseline vs conserving).
Saves: results/nca_dynamics_baseline.gif
       results/nca_dynamics_conserving.gif
       results/nca_dynamics_comparison.gif  (side-by-side)
"""
import argparse
import os
import numpy as np
import tensorflow as tf
import imageio
from PIL import Image, ImageDraw, ImageFont
import inspect

from critical_nca import CriticalNCA
import utils
from evaluate_criticality import apply_conservation

tf.config.set_visible_devices([], 'GPU')

SCALE = 6          # pixels per cell (upscale for visibility)
FPS   = 10         # frames per second in output GIF
N_TIMESTEPS = 100  # how many steps to animate
WIDTH = 100        # default, overridden by args

def load_checkpoint(logdir, ckpt_name=None):
    """Load NCA weights from a checkpoint directory.
    If ckpt_name is given (e.g. '000483_3.5675481.ckpt'), load that explicitly.
    Otherwise falls back to the 'checkpoint' metadata file.
    """
    if ckpt_name is None:
        ckpt_file = os.path.join(logdir, "checkpoint")
        with open(ckpt_file) as f:
            first_line = f.readline()
            start_idx = first_line.find(": ")
            ckpt_name = first_line[start_idx+3:-2]

    args_filename = os.path.join(logdir, "args.json")
    args = utils.ArgsIO(args_filename)
    args.log_dir = logdir

    # Override width/timesteps for animation
    args.ca_width     = WIDTH
    args.ca_timesteps = N_TIMESTEPS

    nca = CriticalNCA()
    sig = inspect.signature(nca.dmodel.load_weights)
    if 'options' in sig.parameters:
        nca.load_weights(os.path.join(logdir, ckpt_name), options=None)
    else:
        nca.load_weights(os.path.join(logdir, ckpt_name))
    return nca, args


def run_nca_frames(nca, args, n_steps):
    """Run NCA forward pass and return list of (width,) state arrays."""
    width = args.ca_width
    x = np.zeros((1, width, nca.channel_n), dtype=np.float32)
    np.random.seed(1)
    x[:, :, :1] = np.random.randint(2, size=(1, width, 1))  # random binary init

    frames = [x[0, :, 0].copy()]
    for t in range(n_steps - 1):
        x = nca(x)
        x = apply_conservation(x, args)
        frames.append(np.array(x)[0, :, 0].copy())
    return frames


def state_to_img(state, width, scale, label=None, frame_height=80):
    """Convert 1D state array to a scaled RGB PIL image with optional label."""
    arr = np.clip(state, 0.0, 1.0)
    row = (arr * 255).astype(np.uint8)
    strip = np.tile(row[np.newaxis, :], (frame_height, 1))
    img = Image.fromarray(strip).resize((width * scale, frame_height), Image.NEAREST).convert("RGB")
    if label:
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), label, fill=(255, 80, 80))
    return img


def frames_to_spacetime(frame_list, scale):
    """Stack 1D frames vertically into a 2D space-time image."""
    rows = [np.clip(f, 0, 1) for f in frame_list]
    arr2d = np.stack(rows, axis=0)                 # (T, W)
    arr2d = (arr2d * 255).astype(np.uint8)
    # Upscale
    h, w = arr2d.shape
    img = Image.fromarray(arr2d).resize((w*scale, h*scale), Image.NEAREST)
    return img.convert("RGB")


BORDER_A = (255, 150, 50)   # orange  — Baseline
BORDER_B = (80, 200, 255)   # cyan    — Mass-Conserving
BORDER_W = 4               # border thickness in pixels


def add_border(img, color, width=BORDER_W):
    """Draw a solid-color rectangle border around a PIL image."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for i in range(width):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color)
    return img


def make_gif(frame_images, path, fps):
    """Save list of PIL images as animated GIF."""
    frame_images[0].save(
        path,
        save_all=True,
        append_images=frame_images[1:],
        loop=0,
        duration=int(1000 / fps)
    )
    print(f"Saved: {path}")


def make_comparison_gif(frames_a, frames_b, label_a, label_b, path, fps, scale, frame_height=80,
                        bg_color=(30, 30, 30)):
    """Side-by-side animated GIF."""
    assert len(frames_a) == len(frames_b)
    width_a = len(frames_a[0])
    width_b = len(frames_b[0])
    gap = scale * 10
    total_w = width_a * scale + gap + width_b * scale
    combined_frames = []
    for fa, fb in zip(frames_a, frames_b):
        img = Image.new("RGB", (total_w, frame_height), color=bg_color)
        arr_a = (np.clip(fa, 0, 1) * 255).astype(np.uint8)
        strip_a = np.tile(arr_a[np.newaxis, :], (frame_height, 1))
        img_a = Image.fromarray(strip_a).resize((width_a * scale, frame_height), Image.NEAREST).convert("RGB")
        arr_b = (np.clip(fb, 0, 1) * 255).astype(np.uint8)
        strip_b = np.tile(arr_b[np.newaxis, :], (frame_height, 1))
        img_b = Image.fromarray(strip_b).resize((width_b * scale, frame_height), Image.NEAREST).convert("RGB")
        add_border(img_a, BORDER_A)
        add_border(img_b, BORDER_B)
        img.paste(img_a, (0, 0))
        img.paste(img_b, (width_a * scale + gap, 0))
        draw = ImageDraw.Draw(img)
        draw.text((BORDER_W + 4, BORDER_W + 4), label_a, fill=BORDER_A)
        draw.text((width_a * scale + gap + BORDER_W + 4, BORDER_W + 4), label_b, fill=BORDER_B)
        combined_frames.append(img)
    make_gif(combined_frames, path, fps)


def make_spacetime_comparison(frames_a, frames_b, label_a, label_b, path, scale):
    """Side-by-side space-time static image."""
    img_a = frames_to_spacetime(frames_a, scale)
    img_b = frames_to_spacetime(frames_b, scale)
    gap = scale * 10
    total_w = img_a.width + gap + img_b.width
    combined = Image.new("RGB", (total_w, img_a.height), color=(30, 30, 30))
    add_border(img_a, BORDER_A)
    add_border(img_b, BORDER_B)
    combined.paste(img_a, (0, 0))
    combined.paste(img_b, (img_a.width + gap, 0))
    draw = ImageDraw.Draw(combined)
    draw.text((BORDER_W + 4, BORDER_W + 4), label_a, fill=BORDER_A)
    draw.text((img_a.width + gap + BORDER_W + 4, BORDER_W + 4), label_b, fill=BORDER_B)
    combined.save(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_logdir",   required=True,
                        help="Log dir of best baseline checkpoint")
    parser.add_argument("--conserving_logdir", required=True,
                        help="Log dir of best conserving checkpoint")
    parser.add_argument("--baseline_ckpt",     default=None,
                        help="Explicit baseline checkpoint name (e.g. 000483_3.5675481.ckpt)")
    parser.add_argument("--conserving_ckpt",   default=None,
                        help="Explicit conserving checkpoint name (e.g. 000312_3.9957634.ckpt)")
    parser.add_argument("--width",      default=100, type=int)
    parser.add_argument("--timesteps",  default=100, type=int)
    parser.add_argument("--fps",          default=10,  type=int)
    parser.add_argument("--scale",        default=6,   type=int)
    parser.add_argument("--frame_height", default=600, type=int,
                        help="Height in pixels of each GIF frame")
    p = parser.parse_args()

    WIDTH       = p.width
    N_TIMESTEPS = p.timesteps
    FPS         = p.fps
    SCALE       = p.scale

    os.makedirs("results", exist_ok=True)

    print("Loading baseline checkpoint...")
    nca_base, args_base = load_checkpoint(p.baseline_logdir, p.baseline_ckpt)
    args_base.conserve = False

    print("Loading conserving checkpoint...")
    nca_cons, args_cons = load_checkpoint(p.conserving_logdir, p.conserving_ckpt)
    args_cons.conserve = True

    print(f"Running {N_TIMESTEPS} timesteps for baseline...")
    frames_base = run_nca_frames(nca_base, args_base, N_TIMESTEPS)

    print(f"Running {N_TIMESTEPS} timesteps for conserving...")
    frames_cons = run_nca_frames(nca_cons, args_cons, N_TIMESTEPS)

    FH = p.frame_height

    # Individual animated GIFs
    gif_base = [state_to_img(f, WIDTH, SCALE, "Baseline",   FH) for f in frames_base]
    gif_cons = [state_to_img(f, WIDTH, SCALE, "Conserving", FH) for f in frames_cons]
    make_gif(gif_base, "results/nca_dynamics_baseline.gif", FPS)
    make_gif(gif_cons, "results/nca_dynamics_conserving.gif", FPS)

    # Side-by-side animated GIF — dark background
    make_comparison_gif(frames_base, frames_cons,
                        "Baseline NCA", "Mass-Conserving NCA",
                        "results/nca_dynamics_comparison.gif", FPS, SCALE, FH)

    # Side-by-side animated GIF — white background (for Google Doc / print)
    make_comparison_gif(frames_base, frames_cons,
                        "Baseline NCA", "Mass-Conserving NCA",
                        "results/nca_dynamics_comparison_white.gif", FPS, SCALE, FH,
                        bg_color=(255, 255, 255))

    # Static space-time comparison image
    make_spacetime_comparison(frames_base, frames_cons,
                              "Baseline NCA", "Mass-Conserving NCA",
                              "results/nca_spacetime_comparison.png", SCALE)

    print("\nDone. Outputs:")
    print("  results/nca_dynamics_baseline.gif")
    print("  results/nca_dynamics_conserving.gif")
    print("  results/nca_dynamics_comparison.gif")
    print("  results/nca_dynamics_comparison_white.gif")
    print("  results/nca_spacetime_comparison.png")
