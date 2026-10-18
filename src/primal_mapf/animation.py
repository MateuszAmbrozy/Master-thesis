"""Render recorded policy trajectories as GIFs and optional MP4 videos."""
import json
import colorsys
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .provenance import file_sha256
from .solvers.api import valid_plan

COLORS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626", "#0891b2", "#ca8a04", "#db2777"]


def validate_record(record: dict) -> None:
    grid = np.asarray(record["obstacles"])
    if grid.ndim != 2 or not np.isin(grid, [0, 1]).all():
        raise ValueError("Obstacle map must be a binary grid")
    starts, goals, trace = record["starts"], record["goals"], record["trace"]
    if not trace or len(starts) != record["agents"] or len(goals) != len(starts):
        raise ValueError("Trace and agent counts do not match")
    if not valid_plan([goals], grid, goals, goals):
        raise ValueError("Goal coordinates must be distinct free cells")
    if not valid_plan(trace, grid, starts, trace[-1]):
        raise ValueError("Cannot animate an invalid trajectory")
    success = list(map(tuple, trace[-1])) == list(map(tuple, goals))
    if success != record["success"]:
        raise ValueError("Recorded success disagrees with terminal positions")
    if record["makespan"] != (len(trace)-1 if success else -1):
        raise ValueError("Recorded makespan disagrees with the trajectory")


def sampled_steps(length: int, max_frames: int) -> list[int]:
    if length < 1 or max_frames < 2:
        raise ValueError("Trace must be non-empty and max_frames must be at least two")
    stride = max(1, math.ceil((length-1)/(max_frames-1)))
    steps = list(range(0, length, stride))
    if steps[-1] != length-1:
        steps.append(length-1)
    return steps


def render_animation(record: dict, output: Path, fps=12, max_frames=240, width=1200,
                     mp4=False) -> dict:
    """Render the actual recorded states; never modify or solve their paths."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError('Install animation dependencies: pip install -e ".[visualization]"') from exc
    validate_record(record)
    if type(fps) is not int or not 1 <= fps <= 50 or width < 480:
        raise ValueError("fps must be in 1..50 and width must be >= 480")
    indices = sampled_steps(len(record["trace"]), max_frames)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Animation output must be empty")
    if mp4 and shutil.which("ffmpeg") is None:
        raise RuntimeError("MP4 export requires ffmpeg; omit --mp4 to save a GIF only")
    output.mkdir(parents=True, exist_ok=True)
    grid = np.asarray(record["obstacles"], dtype=int)
    height, columns = grid.shape
    cell = max(3, min(18, (width-48)//columns, 720//height))
    canvas_width = max(720, columns*cell+48)
    top, left = 112, (canvas_width-columns*cell)//2
    footer_height = max(82, 52+18*math.ceil(record["agents"]/10))
    canvas_height = top+height*cell+footer_height
    colors = COLORS[:]
    while len(colors) < record["agents"]:
        rgb = colorsys.hsv_to_rgb((len(colors)*.61803398875)%1, .8, .8)
        colors.append("#"+"".join(f"{round(v*255):02x}" for v in rgb))

    def font(size, bold=False):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    title_font, label_font, small_font = font(23, True), font(15), font(12)
    background = Image.new("RGB", (canvas_width, canvas_height), "#f8fafc")
    draw = ImageDraw.Draw(background)
    variant = "Improved" if record["variant"] == "improved" else "Classic"
    weights = "fine-tuned" if "fine_tuned" in Path(record["checkpoint"]).parts else "checkpoint"
    draw.text((24,16), f"{variant} PRIMAL / {weights}", font=title_font, fill="#0f172a")
    draw.text((24,51), f"{record['map_name'].capitalize()}  |  {record['agents']} agents  |  seed {record['seed']}",
              font=label_font, fill="#475569")
    for r in range(height):
        for c in range(columns):
            x, y = left+c*cell, top+r*cell
            draw.rectangle((x,y,x+cell-1,y+cell-1), fill="#334155" if grid[r,c] else "#ffffff")
    if cell >= 10:
        for r in range(height+1):
            draw.line((left,top+r*cell,left+columns*cell,top+r*cell), fill="#e2e8f0")
        for c in range(columns+1):
            draw.line((left+c*cell,top,left+c*cell,top+height*cell), fill="#e2e8f0")

    def center(position):
        r, c = position
        return left+c*cell+cell/2, top+r*cell+cell/2

    radius = max(2, int(cell*.38))
    for i, goal in enumerate(record["goals"]):
        x, y = center(goal)
        draw.rectangle((x-radius,y-radius,x+radius,y+radius), outline=colors[i], width=2)
    footer = top+height*cell+14
    for i in range(record["agents"]):
        x = 26+(i%10)*65
        y = footer+(i//10)*18
        draw.ellipse((x,y,x+10,y+10), fill=colors[i])
        draw.text((x+15,y-2), str(i+1), font=small_font, fill="#334155")
    stride = indices[1]-indices[0] if len(indices)>1 else 1
    draw.text((24,canvas_height-37), "Circles: agents   Squares: goals   "
              f"{record['execution']} / {record['environment_profile']}", font=small_font, fill="#475569")
    draw.text((24,canvas_height-20), f"Playback: every {stride} timestep(s), {fps} frames/s",
              font=small_font, fill="#64748b")

    def frame(step):
        image = background.copy()
        painter = ImageDraw.Draw(image)
        positions = record["trace"][step]
        reached = sum(tuple(p)==tuple(g) for p,g in zip(positions, record["goals"]))
        painter.text((24,78), f"Step {step}/{len(record['trace'])-1}    Goals {reached}/{record['agents']}",
                     font=label_font, fill="#0f172a")
        if reached == record["agents"]:
            painter.text((canvas_width-195,78), "All agents on goals", font=label_font, fill="#15803d")
        elif step == len(record["trace"])-1:
            painter.text((canvas_width-195,78), "Time limit reached", font=label_font, fill="#b91c1c")
        painter.rectangle((24,101,canvas_width-24,104), fill="#e2e8f0")
        progress = step/max(1,len(record["trace"])-1)
        painter.rectangle((24,101,24+int((canvas_width-48)*progress),104), fill="#2563eb")
        for i, position in enumerate(positions):
            color = colors[i]
            recent = [center(record["trace"][t][i]) for t in range(max(0,step-5),step+1)]
            if len(recent)>1:
                painter.line(recent, fill=color, width=max(1,cell//5))
            x, y = center(position)
            r = max(2,int(cell*.32))
            painter.ellipse((x-r-1,y-r-1,x+r+1,y+r+1), fill="white")
            painter.ellipse((x-r,y-r,x+r,y+r), fill=color)
            if cell >= 15:
                painter.text((x,y),str(i+1),font=font(9,True),fill="white",anchor="mm")
        return image

    # A shared palette keeps wall and agent colors consistent between frames.
    palette = background.copy()
    swatches = ImageDraw.Draw(palette)
    for i, color in enumerate(colors+["#15803d", "#b91c1c"]):
        swatches.rectangle((i*20,0,i*20+19,19), fill=color)
    palette = palette.quantize(colors=128)
    frames = [frame(step).quantize(palette=palette, dither=Image.Dither.NONE) for step in indices]
    durations = [max(20,round(1000/fps/10)*10)]*len(frames)
    durations[0] = 1000
    durations[-1] = 2500
    gif = output/"animation.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    frame(indices[-1]).save(output/"poster.png")
    (output/"trajectory.json").write_text(json.dumps(record)+"\n")
    details = dict(fps=fps, max_frames=max_frames, frame_steps=indices,
                   gif_sha256=file_sha256(gif), gif_bytes=gif.stat().st_size,
                   width=canvas_width, height=canvas_height)
    if mp4:
        video = output/"animation.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(gif),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(video)], check=True)
        details["mp4_sha256"] = file_sha256(video)
    (output/"render.json").write_text(json.dumps(details, indent=2)+"\n")
    return details
