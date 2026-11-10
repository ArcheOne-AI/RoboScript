"""Set-of-Mark: numbered outlines of proposals on an image and an attribute card per mark, so a
model can pick objects by number (no model calls here); numbered boxes and points, for answers that
come as boxes or points; and a sheet of candidates cut out and enlarged side by side, for a model to
compare look-alike objects.

A mark's number sits on its object where one fits: the deepest point of its mask, the image
border counting as background (a label on an object cut by the border is not cut too), clear of
the other labels and covering little of other objects; else just outside the object's box, with a
line to it; smallest objects first, so they get the free spots.

Attributes come from geometry and pixels, never from a model: the nearest basic color to the
mark's median color (no white balance), footprint and height from depth, and where
the mark sits as the robot sees it (x to its right, y away from it): relative to the table center
when every table edge is known, else relative to the robot's midline and base.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from roboscript.perception.proposals import Proposal
from roboscript.perception.table import Table

BASIC_COLORS = {
    "red": (200, 30, 30),
    "orange": (230, 130, 30),
    "yellow": (220, 210, 40),
    "green": (40, 170, 60),
    "cyan": (40, 190, 200),
    "blue": (40, 60, 200),
    "purple": (130, 50, 170),
    "pink": (230, 130, 180),
    "brown": (120, 75, 40),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "black": (25, 25, 25),
}
_EIGHT = np.ones((3, 3), dtype=bool)


def color_name(pixels: ArrayLike) -> str:
    """The basic color nearest to the median of RGB pixels (..., 3)."""
    median = np.median(np.asarray(pixels, dtype=float).reshape(-1, 3), axis=0)
    names = list(BASIC_COLORS)
    return names[int(np.argmin([np.linalg.norm(median - BASIC_COLORS[n]) for n in names]))]


def mark_anchor(mask: ArrayLike) -> tuple[int, int]:
    """A pixel (u, v) deep inside the mask: the maximum of its distance transform, the image
    border counting as background."""
    distance = _depth(np.asarray(mask, dtype=bool))
    v, u = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(u), int(v)


Box = tuple[int, int, int, int]


def place_labels(
    masks: Mapping[int, NDArray[np.bool_]], size: tuple[int, int], shape: tuple[int, int]
) -> dict[int, tuple[tuple[int, int], tuple[int, int] | None]]:
    """Where each mark's label (`size`: width, height) goes: its center and, when it sits outside
    its object, the object's pixel its line goes to (the module's rule)."""
    width, height = size
    others = np.zeros(shape, dtype=np.int32)
    for mask in masks.values():
        others += mask
    placed: list[Box] = []
    out: dict[int, tuple[tuple[int, int], tuple[int, int] | None]] = {}
    for number in sorted(masks, key=lambda n: int(masks[n].sum())):
        mask = masks[number]
        foreign = (others - mask) > 0
        found: tuple[tuple[int, int], tuple[int, int] | None] | None = None
        distance = _depth(mask)
        vs, us = np.nonzero(distance >= min(size) / 2)
        for k in np.argsort(-distance[vs, us])[:CANDIDATES]:
            center = (int(us[k]), int(vs[k]))
            box = _box(center, size)
            if _free(box, shape, placed) and _covers(foreign, box) <= FOREIGN * width * height:
                found = (center, None)
                break
        if found is None and mask.any():
            v, u = np.nonzero(mask)
            cu, cv = int(np.median(u)), int(np.median(v))
            u0, v0, u1, v1 = int(u.min()), int(v.min()), int(u.max()), int(v.max())
            gap_u, gap_v = width // 2 + 4, height // 2 + 4
            spots = [
                (cu, v0 - gap_v), (cu, v1 + gap_v), (u0 - gap_u, cv), (u1 + gap_u, cv),
                (u0 - gap_u, v0 - gap_v), (u1 + gap_u, v0 - gap_v),
                (u0 - gap_u, v1 + gap_v), (u1 + gap_u, v1 + gap_v),
            ]  # fmt: skip
            for spot in spots:
                box = _box(spot, size)
                if (
                    _free(box, shape, placed)
                    and _covers(others > 0, box) <= FOREIGN * width * height
                ):
                    nearest = int(np.argmin(np.abs(u - spot[0]) + np.abs(v - spot[1])))
                    found = (spot, (int(u[nearest]), int(v[nearest])))
                    break
        if found is None:  # crowded: the deepest point, kept inside the image
            u, v = mark_anchor(mask)
            u = int(np.clip(u, width // 2 + MARGIN, shape[1] - width // 2 - MARGIN))
            v = int(np.clip(v, height // 2 + MARGIN, shape[0] - height // 2 - MARGIN))
            found = ((u, v), None)
        placed.append(_box(found[0], size))
        out[number] = found
    return out


CANDIDATES = 400
"""Deepest points of a mask tried for its label."""
FOREIGN = 0.1
"""A label may cover this share of its box with other objects' pixels."""
MARGIN = 2
"""Pixels between a label and the image border."""


def som_overlay(
    rgb: NDArray[np.uint8],
    marks: Mapping[int, NDArray[np.bool_]],
    font_size: int = 18,
    labels: Mapping[int, str] | None = None,
) -> tuple[NDArray[np.uint8], dict[int, tuple[int, int]]]:
    """The image with each mask outlined (black with a white core) and numbered (or given its
    label) where `place_labels` puts it, joined by a line to its object when it sits outside it;
    returns the image and each mark's anchor (u, v), a pixel deep inside its object."""
    image = np.array(rgb[..., :3], dtype=np.uint8)
    anchors = {}
    closed = {}
    for number, mask in marks.items():
        closed[number] = mask | ndimage.binary_closing(mask, np.ones((5, 5), dtype=bool))
        edge = closed[number] & ~ndimage.binary_erosion(closed[number], _EIGHT)
        image[ndimage.binary_dilation(edge, _EIGHT)] = 0
        image[edge] = 255
        anchors[number] = mark_anchor(closed[number])
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=font_size)
    texts = {number: (labels or {}).get(number, str(number)) for number in marks}
    sizes = [_text_size(draw, font, text) for text in texts.values()]
    size = (max((w for w, _ in sizes), default=0), max((h for _, h in sizes), default=0))
    for number, (center, target) in place_labels(closed, size, image.shape[:2]).items():
        if target is not None:
            for width, color in ((4, (0, 0, 0)), (2, (255, 255, 255))):
                draw.line((*center, *target), fill=color, width=width)
        _tag(draw, font, texts[number], *center)
    return np.asarray(canvas), anchors


def crop_sheet(
    candidates: Sequence[Sequence[tuple[NDArray[np.uint8], NDArray[np.bool_]]]],
    letters: Sequence[str],
    side: int = 288,
    font_size: int = 22,
) -> NDArray[np.uint8]:
    """Candidates side by side, each under its letter: every view given for it (an image and the
    candidate's mask there), cut out around the mask with a margin, outlined and enlarged to fit
    `side` pixels, one below the other."""
    font = ImageFont.load_default(size=font_size)
    header = font_size + 12
    rows = max((len(views) for views in candidates), default=1)
    height = header + (side + GAP) * rows
    tiles = []
    for views, letter in zip(candidates, letters, strict=True):
        tile = Image.fromarray(np.full((height, side + 2 * GAP, 3), 255, dtype=np.uint8))
        for row, (rgb, mask) in enumerate(views):
            crop = _crop(rgb, mask, side)
            left = GAP + (side - crop.width) // 2
            top = header + row * (side + GAP) + (side - crop.height) // 2
            tile.paste(crop, (left, top))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, side + 2 * GAP - 1, height - 1), outline=(0, 0, 0), width=2)
        draw.text((GAP + 4, 4), letter, fill=(0, 0, 0), font=font)
        tiles.append(np.asarray(tile))
    return np.concatenate(tiles, axis=1)


GAP = 6
"""Pixels around each cut-out candidate on a sheet."""


def _crop(rgb: NDArray[np.uint8], mask: NDArray[np.bool_], side: int) -> Image.Image:
    """The image around a mask (its box and 80 % more, at least 24 pixels each way), the mask
    outlined, enlarged or shrunk to fit `side` pixels."""
    v, u = np.nonzero(mask)
    h, w = mask.shape
    cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
    half = max(u.max() - u.min(), v.max() - v.min()) * 0.8 + 24
    u0, u1 = int(max(0, cu - half)), int(min(w, cu + half))
    v0, v1 = int(max(0, cv - half)), int(min(h, cv + half))
    image = np.array(rgb[v0:v1, u0:u1, :3], dtype=np.uint8)
    part = mask[v0:v1, u0:u1]
    edge = part & ~ndimage.binary_erosion(part, _EIGHT)
    image[ndimage.binary_dilation(edge, _EIGHT)] = 0
    image[edge] = 255
    scale = side / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)


def _depth(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Each pixel's distance to the mask's outside, the image border counting as outside."""
    distance: NDArray[np.float64] = ndimage.distance_transform_edt(np.pad(mask, 1))
    return distance[1:-1, 1:-1]


def _box(center: tuple[int, int], size: tuple[int, int]) -> Box:
    (u, v), (w, h) = center, size
    return u - w // 2, v - h // 2, u - w // 2 + w, v - h // 2 + h


def _free(box: Box, shape: tuple[int, ...], placed: list[Box]) -> bool:
    """Inside the image, off every label placed."""
    u0, v0, u1, v1 = box
    inside = u0 >= MARGIN and v0 >= MARGIN and u1 <= shape[1] - MARGIN and v1 <= shape[0] - MARGIN
    return inside and not any(
        not (u1 <= a0 or a1 <= u0 or v1 <= b0 or b1 <= v0) for a0, b0, a1, b1 in placed
    )


def _covers(pixels: NDArray[np.bool_], box: Box) -> int:
    u0, v0, u1, v1 = box
    return int(pixels[max(v0, 0) : v1, max(u0, 0) : u1].sum())


def _text_size(
    draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, text: str
) -> tuple[int, int]:
    """The size of `_tag`'s box around `text`."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return int(right - left) + 6, int(bottom - top) + 6


def box_overlay(
    rgb: NDArray[np.uint8],
    boxes: Mapping[int, Sequence[float]],
    points: Mapping[int, Sequence[float]] | None = None,
    font_size: int = 16,
) -> NDArray[np.uint8]:
    """The image with each box (u0, v0, u1, v1) outlined and numbered at its top-left corner and
    each point (u, v) crossed and numbered (continuous pixel coordinates)."""
    canvas = Image.fromarray(np.array(rgb[..., :3], dtype=np.uint8))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=font_size)
    for number, (u0, v0, u1, v1) in boxes.items():
        draw.rectangle((u0, v0, u1 - 1, v1 - 1), outline=(0, 0, 0), width=3)
        draw.rectangle((u0, v0, u1 - 1, v1 - 1), outline=(255, 255, 255), width=1)
        _tag(draw, font, str(number), int(u0) + 8, int(v0) + 8)
    for number, (u, v) in (points or {}).items():
        for width, color in ((4, (0, 0, 0)), (2, (255, 255, 255))):
            draw.line((u - 6, v - 6, u + 6, v + 6), fill=color, width=width)
            draw.line((u - 6, v + 6, u + 6, v - 6), fill=color, width=width)
        _tag(draw, font, str(number), int(u) + 14, int(v))
    return np.asarray(canvas)


def _tag(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text: str,
    u: int,
    v: int,
) -> None:
    """White text on a black box centered at (u, v)."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left, bottom - top
    x0, y0 = u - w // 2 - 3, v - h // 2 - 3
    draw.rectangle((x0, y0, x0 + w + 6, y0 + h + 6), fill=(0, 0, 0))
    draw.text((x0 + 3 - left, y0 + 3 - top), text, fill=(255, 255, 255), font=font)


def attribute_card(
    rgb: NDArray[np.uint8],
    proposals: Sequence[Proposal],
    table: Table,
    camera: str | None = None,
    base: ArrayLike | None = None,
) -> str:
    """One line per proposal: color (its pixels in `camera`'s image `rgb`, by default the camera
    it was found in), footprint, height and where it is (from its summary): relative to the
    table's center when all four table edges are known; else, since the middle of a partly seen
    table is skewed, relative to the robot, left or right of its midline and how far ahead of its
    base (`base`, the robot base's world x, y); without a base, relative to the middle of what is
    seen of the table."""
    known = all(edge is not None for edge in table.edges.values())
    lines = []
    for p in proposals:
        mask = p.masks[camera or p.camera]
        s = p.summary
        if known or base is None:
            cx, cy = table.middle
            x, y = s.center[0] - cx, s.center[1] - cy
            side = "left" if x < -SIDE else "right" if x > SIDE else "middle"
            depth = "near" if y < -DEPTH else "far" if y > DEPTH else "center"
            where = (
                f"table position {side}/{depth} (x={x * 100:+.0f} cm to the robot's right, "
                f"y={y * 100:+.0f} cm away from the robot, from the table's center)"
            )
        else:
            bx, by = np.asarray(base, dtype=float)[:2]
            x, ahead = s.center[0] - bx, s.center[1] - by
            side = "left of" if x < -SIDE else "right of" if x > SIDE else "on"
            depth = "near" if ahead < NEAR else "far" if ahead > FAR else "middle distance"
            where = (
                f"{side} the robot's midline, {depth} (x={x * 100:+.0f} cm to the robot's "
                f"right, {ahead * 100:.0f} cm ahead of the robot's base)"
            )
        lines.append(
            f"mark {p.id}: {color_name(rgb[mask])}, {s.size[0] * 100:.1f} x {s.size[1] * 100:.1f}"
            f" cm footprint, {s.size[2] * 100:.1f} cm tall, {where}"
        )
    return "\n".join(lines)


SIDE = 0.05
"""Within this of the reference in x a mark is in the middle (on the midline)."""
DEPTH = 0.1
"""Within this of the table's center in y a mark is at the center."""
NEAR = 0.55
FAR = 0.75
"""Closer ahead of the robot's base than `NEAR` a mark is near, farther than `FAR` far."""
