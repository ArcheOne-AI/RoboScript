"""Review pages of the demonstration choice (a human checks the pick).

Per task, the top candidates side by side: the episode video, the ranking criteria, the keypoint
table and head-camera frames with the keypoints drawn (the arrow runs along the gripper's pointing
axis to the TCP; a filled dot is a closed gripper). Choosing another episode is stored in the
browser; the index page exports the choices as `reviewed.json` for `select_demos.py
--reviewed`.
"""

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from roboscript.contracts.robot import Arm
from roboscript.demos import Demo, Keypoint, KeypointSet
from roboscript.demos.manifest import DemoManifest, TaskDemo
from roboscript.demos.selection import RankedEpisode
from roboscript.spatial.camera import project
from roboscript.spatial.rotation import quat_rotate
from roboscript_robotwin.demos.expert import read_images

CAMERA: Final = "head_camera"
SCALE: Final = 2  # frames are drawn at twice the camera's resolution
AXIS_M: Final = 0.06
COLOURS: Final[dict[Arm, tuple[int, int, int]]] = {"left": (255, 120, 0), "right": (0, 140, 255)}
FOCUS: Final = (255, 230, 0)


@dataclass(frozen=True)
class Shown:
    """A rendered candidate: its ranking, keypoints and files under the task directory."""

    ranked: RankedEpisode
    keypoints: KeypointSet
    directory: str
    frames: list[tuple[str, str]]  # (image file, caption)


def render_candidate(demo: Demo, keypoints: KeypointSet, directory: Path) -> list[tuple[str, str]]:
    """Writes `overview.jpg` (the first frame with both arms' paths) and one image per keypoint
    into `directory`; returns (file, caption) pairs, the overview first."""
    directory.mkdir(parents=True, exist_ok=True)
    focus: list[tuple[Arm, Keypoint] | None] = [None]
    focus += [(arm, kp) for arm, kps in keypoints.arms.items() for kp in kps]
    frames = [0 if f is None else f[1].frame for f in focus]
    calib = demo.calib[CAMERA]
    paths = {arm: t.tcp for arm, t in demo.arms.items()}
    out = []
    images = read_images(demo, CAMERA, frames)
    for image, t, f in zip(images, frames, focus, strict=True):
        img = draw(image, calib.K[t], calib.cam2world[t], keypoints, None if f else paths, f)
        if f is None:
            name, caption = "overview.jpg", "paths"
        else:
            arm, kp = f
            name = f"{arm}_K{kp.index}.jpg"
            caption = f"{arm} K{kp.index} {kp.phase} ({kp.cue}), frame {t}"
        img.save(directory / name, quality=88)
        out.append((name, caption))
    return out


def draw(
    image: NDArray[np.uint8],
    K: NDArray[np.float64],
    cam2world: NDArray[np.float64],
    keypoints: KeypointSet,
    paths: dict[Arm, NDArray[np.float64]] | None = None,
    focus: tuple[Arm, Keypoint] | None = None,
) -> Image.Image:
    img = Image.fromarray(image).convert("RGB")
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.Resampling.BICUBIC)
    K = K.copy()
    K[:2] *= SCALE
    pen = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    for arm, tcp in (paths or {}).items():
        uv, _, front = project(tcp[:, :3], K, cam2world)
        pen.line([tuple(p) for p in uv[front].tolist()], fill=COLOURS[arm], width=1)
    for arm, kps in keypoints.arms.items():
        for kp in kps:
            tip = np.asarray(kp.tcp.p)
            tail = tip - AXIS_M * quat_rotate(kp.tcp.q, [1.0, 0.0, 0.0])
            (tu, tv), (u, v) = project(np.stack([tail, tip]), K, cam2world)[0].tolist()
            focused = focus is not None and focus[0] == arm and focus[1] is kp
            colour, r = COLOURS[arm], 7 if focused else 4
            edge = FOCUS if focused else colour
            pen.line([(tu, tv), (u, v)], fill=edge, width=3 if focused else 2)
            fill = colour if kp.gripper == "closed" else None
            pen.ellipse([u - r, v - r, u + r, v + r], outline=edge, fill=fill, width=2)
            pen.text((u + r + 2, v - r - 2), f"{arm[0].upper()}{kp.index}", fill=colour, font=font)
    return img


_STYLE: Final = """
body { font: 14px/1.4 system-ui, sans-serif; margin: 16px; color: #1d2330;
  background: #f7f7f9; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin: 4px 0; }
.cols { display: flex; gap: 12px; align-items: flex-start; overflow-x: auto; }
.col { background: #fff; border: 1px solid #d6d9e0; border-radius: 6px; padding: 10px;
  min-width: 660px; }
.col.chosen { border: 2px solid #2f7d32; }
.thumbs { display: flex; flex-wrap: wrap; gap: 6px; }
figure { margin: 0; width: 320px; } figure img { width: 320px; } figcaption { font-size: 12px; }
table { border-collapse: collapse; font-size: 12px; }
td, th { border: 1px solid #d6d9e0; padding: 2px 6px; text-align: left; }
.pen, .flag { color: #a33; } .flag { font-weight: 600; }
"""

_TASK_SCRIPT: Final = """
const radios = document.getElementsByName("pick");
const show = () => {
  const mine = localStorage.getItem(key);
  document.getElementById("pick").textContent = mine ?? `none (the manifest has ${chosen})`;
  for (const r of radios) r.checked = r.value === (mine ?? chosen);
};
for (const r of radios) r.addEventListener("change", () => {
  localStorage.setItem(key, r.value);
  show();
});
show();
"""

_INDEX_SCRIPT: Final = """
const PREFIX = "rs-demo-pick:";
const picks = () => Object.fromEntries(Object.keys(localStorage)
  .filter((k) => k.startsWith(PREFIX))
  .map((k) => [k.slice(PREFIX.length), Number(localStorage.getItem(k))]));
for (const td of document.querySelectorAll("td.mine")) {
  td.textContent = picks()[td.dataset.task] ?? "";
}
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify(picks(), null, 1)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"),
    { href: URL.createObjectURL(blob), download: "reviewed.json" });
  a.click();
};
document.getElementById("clear").onclick = () => {
  for (const k of Object.keys(localStorage)) {
    if (k.startsWith(PREFIX)) localStorage.removeItem(k);
  }
  location.reload();
};
"""


def task_page(task: str, entry: TaskDemo, shown: list[Shown], ranking: list[RankedEpisode]) -> str:
    cols = "".join(_column(s, entry) for s in shown)
    rows = "".join(
        f"<tr><td><input type=radio name=pick value={r.episode}></td><td>{r.rank}</td>"
        f"<td>{r.episode}</td><td>{r.seed}</td><td>{r.score:.2f}</td>"
        f"<td>{html.escape('; '.join(r.reasons))}</td></tr>"
        for r in ranking
    )
    notes = "".join(f"<li class=flag>{html.escape(n)}</li>" for n in entry.notes)
    head = "<tr><th>pick</th><th>rank</th><th>episode</th><th>seed</th><th>score</th>"
    return f"""<!doctype html><html><head><meta charset=utf-8><title>{task}</title>
<style>{_STYLE}</style></head><body>
<p><a href="../index.html">all tasks</a></p>
<h1>{task}: episode {entry.episode} (seed {entry.seed}), {entry.source}, {entry.status};
automatic pick episode {entry.auto_episode}</h1>
<p>Instruction: {html.escape(entry.instruction)}</p>
<ul>{notes}</ul>
<p>Your pick: <b id=pick></b> (kept in this browser; export it from the index page)</p>
<div class=cols>{cols}</div>
<details><summary>All {len(ranking)} candidates</summary>
<table>{head}<th>criteria</th></tr>{rows}</table></details>
<script>const key = {json.dumps("rs-demo-pick:" + task)}, chosen = "{entry.episode}";
{_TASK_SCRIPT}</script></body></html>
"""


def _column(s: Shown, entry: TaskDemo) -> str:
    r = s.ranked
    roles = [
        ("chosen", r.episode == entry.episode),
        ("automatic pick", r.episode == entry.auto_episode),
    ]
    label = ", ".join(name for name, on in roles if on)
    crit = "".join(
        f"<li>{html.escape(c.note)}"
        + (f" <span class=pen>+{c.cost:.2f}</span>" if c.cost >= 0.005 else "")
        + "</li>"
        for c in r.criteria
    )
    rows = "".join(
        f"<tr><td>{arm}</td><td>K{kp.index}</td><td>{kp.phase}</td><td>{kp.cue}</td>"
        f"<td>{kp.gripper}</td><td>{kp.frame}</td>"
        f"<td>{', '.join(f'{v:+.3f}' for v in kp.tcp.p)}</td></tr>"
        for arm, kps in s.keypoints.arms.items()
        for kp in kps
    )
    figs = "".join(
        f'<figure><img src="{s.directory}/{f}" loading=lazy>'
        f"<figcaption>{html.escape(c)}</figcaption></figure>"
        for f, c in s.frames
    )
    return f"""<div class="col{" chosen" if r.episode == entry.episode else ""}">
<h2><label><input type=radio name=pick value={r.episode}> #{r.rank} episode {r.episode}
(seed {r.seed}), score {r.score:.2f}{f" ({label})" if label else ""}</label></h2>
<video src="{s.directory}/video.mp4" width=640 controls loop muted preload=metadata></video>
<ul>{crit}</ul>
<table><tr><th>arm</th><th>kp</th><th>phase</th><th>cue</th><th>gripper</th><th>frame</th>
<th>TCP xyz (m)</th></tr>{rows}</table>
<div class=thumbs>{figs}</div></div>"""


def index_page(manifest: DemoManifest) -> str:
    rows = "".join(
        f"<tr><td><a href='{t}/review.html'>{t}</a></td><td>{e.category or ''}</td>"
        f"<td>{e.episode}</td><td>{e.seed}</td><td>{e.source}: {e.status}</td>"
        f"<td>{e.score:.2f}</td>"
        f"<td>{'' if e.margin is None else f'{e.margin:.2f}'}</td>"
        f"<td class=flag>{html.escape(' / '.join(e.notes))}</td>"
        f"<td>{' / '.join(f'{a} {n}' for a, n in e.keypoint_counts.items())}</td>"
        f"<td data-task='{t}' class=mine></td></tr>"
        for t, e in manifest.tasks.items()
    )
    head = "<tr><th>task</th><th>category</th><th>episode</th><th>seed</th><th>status</th>"
    return f"""<!doctype html><html><head><meta charset=utf-8><title>Demo choice</title>
<style>{_STYLE}</style></head><body>
<h1>Demonstration per task ({len(manifest.tasks)} tasks)</h1>
<p>Source {html.escape(manifest.source_root)}; categories from {manifest.categories}; updated
{manifest.created}. Open a task to confirm
or change its pick, then <button id=export>download reviewed.json</button>
<button id=clear>forget my picks</button></p>
<table>{head}<th>score</th><th>margin</th><th>notes</th><th>keypoints</th><th>your pick</th></tr>
{rows}</table>
<script>{_INDEX_SCRIPT}</script></body></html>
"""
