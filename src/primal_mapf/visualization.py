"""Deterministic SVG trajectory rendering, independent of model checkpoints."""
from html import escape
from pathlib import Path


def trajectory_svg(obstacles, starts, goals, trace, title="MAPF trajectory", animate=False) -> str:
    scale, margin = 36, 24
    height, width = obstacles.shape
    w, h = width*scale + 2*margin, height*scale + 2*margin + 52
    def point(position):
        r, c = position
        return margin+c*scale+scale/2, margin+r*scale+scale/2
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">',
             f'<title>{escape(title)}</title>', '<rect width="100%" height="100%" rx="12" fill="#f8fafc"/>']
    for r in range(height):
        for c in range(width):
            color = '#334155' if obstacles[r,c] else '#ffffff'
            lines.append(f'<rect x="{margin+c*scale}" y="{margin+r*scale}" width="{scale}" height="{scale}" fill="{color}" stroke="#e2e8f0"/>')
    palette = ['#2563eb', '#dc2626', '#059669', '#9333ea', '#d97706', '#0891b2']
    for i in range(len(starts)):
        color = palette[i % len(palette)]
        points = ' '.join(f'{x},{y}' for x,y in [point(step[i]) for step in trace])
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" opacity=".45"/>')
        gx, gy = point(goals[i])
        lines.append(f'<circle cx="{gx}" cy="{gy}" r="12" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="3 2"/>')
        sx, sy = point(starts[i])
        lines.append(f'<circle cx="{sx}" cy="{sy}" r="9" fill="{color}">')
        if animate and len(trace) > 1:
            xs = ';'.join(str(point(step[i])[0]) for step in trace)
            ys = ';'.join(str(point(step[i])[1]) for step in trace)
            for attr, values in [('cx', xs), ('cy', ys)]:
                lines.append(f'<animate attributeName="{attr}" values="{values}" dur="{max(4, len(trace)/2)}s" repeatCount="indefinite"/>')
        lines.append('</circle>')
        lines.append(f'<text x="{sx}" y="{sy+4}" text-anchor="middle" fill="white" font-family="sans-serif" font-size="11">{i+1}</text>')
    lines.append(f'<text x="{margin}" y="{h-30}" font-family="sans-serif" font-size="14" fill="#0f172a">{escape(title)}</text>')
    lines.append(f'<text x="{margin}" y="{h-11}" font-family="sans-serif" font-size="11" fill="#475569">Starts: solid · goals: dashed · steps: {len(trace)-1}</text></svg>')
    return '\n'.join(lines)


def write_svg(path: Path, *args, **kwargs) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trajectory_svg(*args, **kwargs))
