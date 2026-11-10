"""The vision service over a real socket, with color-keyed numpy stand-ins for the models."""

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import anyio
import httpx
import numpy as np
import pytest
from numpy.typing import NDArray

from roboscript.blobs import LocalBlobStore
from roboscript.contracts import versions
from roboscript.contracts.blob import BlobRef
from roboscript.contracts.errors import InfraError, ProtocolError
from roboscript.contracts.ids import BlobId
from roboscript.contracts.vision import TIMEOUTS, Detection, SegmentPrompt, Vision
from roboscript.rpc import Ready, ServiceFault, aclose, client, read_ready
from roboscript_vision.config import VisionConfig
from roboscript_vision.service import VisionService, run

RED, BLUE = (220, 30, 30), (30, 40, 210)
COLORS = {"red": RED, "blue": BLUE}


def scene() -> NDArray[np.uint8]:
    """120 x 160, gray, a red rectangle at rows 20-49, columns 10-59 and a blue one at rows
    70-99, columns 90-149."""
    image = np.full((120, 160, 3), 128, np.uint8)
    image[20:50, 10:60] = RED
    image[70:100, 90:150] = BLUE
    return image


class ColorModel:
    """Finds the colors a text names: their box and mask, then the whole image as a worse second
    guess. A box prompt takes the colored pixels inside it, a point the pixels of its color.
    `load` waits for `gate` (open unless `loaded` is false); every call records its thread. Texts
    holding "slow" wait for `release` first; then those holding "explode" fail as a broken model
    would, and "too long" as a refused request."""

    source = "test/colors"
    revision = "test"

    def __init__(self, name: str, *, loaded: bool = True) -> None:
        self.name = name
        self.gate = threading.Event()
        if loaded:
            self.gate.set()
        self.entered, self.release = threading.Event(), threading.Event()
        self.threads: set[int] = set()
        self.calls = 0

    def load(self) -> dict[str, str]:
        assert self.gate.wait(10)
        self.threads.add(threading.get_ident())
        return {"fake": self.name}

    def detect(self, image: NDArray[np.uint8], query: str, top_k: int) -> list[Detection]:
        self._called(query)
        height, width = image.shape[:2]
        found = []
        if (mask := _color(image, query)).any():
            v, u = np.nonzero(mask)
            box = (u.min(), v.min(), u.max() + 1, v.max() + 1)
            found.append(Detection(box=box, score=0.9, label=query))
        found.append(Detection(box=(0, 0, width, height), score=0.1, label="image"))
        return found[:top_k]

    def segment(
        self, image: NDArray[np.uint8], prompt: SegmentPrompt, top_k: int
    ) -> list[tuple[NDArray[np.bool_], float]]:
        self._called(prompt.text or "")
        if prompt.text is not None:
            mask = _color(image, prompt.text)
            return [(mask, 0.9)][:top_k] if mask.any() else []
        colored = ~np.all(image == 128, axis=-1)
        masks = []
        for u0, v0, u1, v1 in prompt.boxes or []:
            inside = np.zeros(colored.shape, bool)
            inside[int(v0) : int(v1), int(u0) : int(u1)] = True
            masks.append((colored & inside, 0.8))
        for u, v in prompt.points or []:
            masks.append((np.all(image == image[int(v), int(u)], axis=-1), 0.7))
        return masks

    def _called(self, text: str) -> None:
        self.calls += 1
        self.threads.add(threading.get_ident())
        if "slow" in text:
            self.entered.set()
            assert self.release.wait(10)
        if "explode" in text:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        if "too long" in text:
            raise ProtocolError("invalid_request", f"{text!r} is 99 tokens; it reads at most 32")


def _color(image: NDArray[np.uint8], text: str) -> NDArray[np.bool_]:
    rgb = next((c for word, c in COLORS.items() if word in text), None)
    return np.zeros(image.shape[:2], bool) if rgb is None else np.all(image == rgb, axis=-1)


def make_service(tmp_path: Path, max_pending: int = 8, **models: ColorModel) -> VisionService:
    config = VisionConfig(
        blob_dir=tmp_path / "blobs", cache_dir=tmp_path / "hf", max_pending=max_pending
    )
    return VisionService(config, models.get("detector"), models.get("segmenter"))


@asynccontextmanager
async def serving(service: VisionService, tmp_path: Path) -> AsyncIterator[tuple[Ready, Vision]]:
    """The service on a real socket, and a client once it is ready (or, with a gated model,
    as soon as it listens)."""
    ready_file = tmp_path / "ready.json"
    async with anyio.create_task_group() as tg, httpx.AsyncClient(trust_env=False) as http:
        tg.start_soon(partial(run, service, ready_file=ready_file))
        with anyio.fail_after(10):
            while True:
                if (ready := read_ready(ready_file)) is not None:
                    break
                await anyio.sleep(0.01)
        gated = [m for m in (service.detector, service.segmenter) if isinstance(m, ColorModel)]
        if all(m.gate.is_set() for m in gated):
            await wait_ready(http, ready.url)
        proxy = client(Vision, ready.url, token=None, timeout=10, timeouts=TIMEOUTS)
        yield ready, proxy
        await aclose(proxy)
        tg.cancel_scope.cancel()


async def wait_ready(http: httpx.AsyncClient, url: str) -> None:
    with anyio.fail_after(10):
        while True:
            if (await http.get(f"{url}/readyz")).status_code == 200:
                return
            await anyio.sleep(0.01)


async def test_models_load_and_warm_up_after_the_service_listens(tmp_path: Path) -> None:
    detector = ColorModel("detector", loaded=False)
    segmenter = ColorModel("segmenter", loaded=False)
    service = make_service(tmp_path, detector=detector, segmenter=segmenter)
    async with serving(service, tmp_path) as (ready, proxy):
        async with httpx.AsyncClient(trust_env=False) as http:
            assert (await http.get(f"{ready.url}/readyz")).status_code == 503
            image = await service.blobs.aput_array(scene())
            with pytest.raises(InfraError, match="still loading"):
                await proxy.detect(image, ["red"])
            detector.gate.set()
            segmenter.gate.set()
            await wait_ready(http, ready.url)
        info = await proxy.info()
        assert [(m.name, m.revision, m.task) for m in info.models] == [
            ("detector", "test", "detect"),
            ("segmenter", "test", "segment"),
        ]
        assert info.runtime["fake"] == "segmenter"  # merged: the later model's facts win
        assert info.runtime["roboscript_vision"]
        assert (detector.calls, segmenter.calls) == (1, 3)  # warm-up: detect; text, box, point


async def test_segment_returns_masks_by_reference(tmp_path: Path) -> None:
    service = make_service(tmp_path, segmenter=ColorModel("colors"))
    blobs = LocalBlobStore(tmp_path / "blobs")  # the caller's view of the shared directory
    async with serving(service, tmp_path) as (_, proxy):
        image = blobs.put_array(scene())
        red = np.all(scene() == RED, axis=-1)
        blue = np.all(scene() == BLUE, axis=-1)

        (text,) = (await proxy.segment(image, SegmentPrompt(text="red"))).masks
        mask = blobs.get_array(text.mask)
        assert mask.dtype == np.bool_ and mask.shape == (120, 160)
        assert np.array_equal(mask, red)
        assert text.box == (10, 20, 60, 50) and text.score == 0.9  # the mask's box, as detections
        assert (await proxy.segment(image, SegmentPrompt(text="green"))).masks == []  # no match

        boxes = [(80, 60, 160, 120), (0, 0, 70, 60), (60, 0, 80, 10)]
        result = await proxy.segment(image, SegmentPrompt(boxes=boxes))
        masks = [blobs.get_array(m.mask) for m in result.masks]
        assert np.array_equal(masks[0], blue) and np.array_equal(masks[1], red)
        assert not masks[2].any() and result.masks[2].box is None  # an empty mask

        result = await proxy.segment(image, SegmentPrompt(points=[(30.5, 30.5), (100, 80)]))
        masks = [blobs.get_array(m.mask) for m in result.masks]
        assert np.array_equal(masks[0], red) and np.array_equal(masks[1], blue)


async def test_requests_it_cannot_serve_are_refused(tmp_path: Path) -> None:
    service = make_service(tmp_path, segmenter=ColorModel("colors"))
    blobs = service.blobs
    async with serving(service, tmp_path) as (ready, proxy):
        image = await blobs.aput_array(scene())
        depth = await blobs.aput_array(np.zeros((120, 160), np.float32))
        huge = await blobs.aput_array(np.zeros((1081, 1921, 3), np.uint8))
        empty = await blobs.aput_array(np.zeros((0, 160, 3), np.uint8))
        rgba = await blobs.aput_image(np.zeros((120, 160, 4), np.uint8), "png")
        json_blob = await blobs.aput_bytes(b"{}", "application/json")
        not_png = await blobs.aput_bytes(b"not a png", "image/png")
        text = SegmentPrompt(text="a")
        refused = {
            "no grounding_dino model": lambda: proxy.detect(image, ["red"]),
            r"RGB \(H, W, 3\) uint8, blob \w+ is float32 \(120, 160\)": lambda: proxy.segment(
                depth, text
            ),
            r"is RGBA \(120, 160, 4\)": lambda: proxy.segment(rgba, text),
            "is 1921x1081; an image holds 1 to 2073600 pixels": lambda: proxy.segment(huge, text),
            "is 160x0; an image holds 1 to": lambda: proxy.segment(empty, text),
            "an image is one of": lambda: proxy.segment(json_blob, text),
            "not a readable image/png": lambda: proxy.segment(not_png, text),
            "leaves the 160x120 image": lambda: proxy.segment(
                image, SegmentPrompt(boxes=[(100, 100, 170, 110)])
            ),
            "outside the 160x120 image": lambda: proxy.segment(
                image, SegmentPrompt(points=[(160, 10)])
            ),
        }
        for message, call in refused.items():
            with pytest.raises(ProtocolError, match=message) as caught:
                await call()
            assert caught.value.code == "invalid_request"
        missing = BlobRef(id=BlobId("f" * 64), size=1, media_type="application/x-npy")
        with pytest.raises(ProtocolError, match="no blob") as caught:
            await proxy.segment(missing, text)
        assert caught.value.code == "not_found"

        # The server checks too, not only our client: two kinds, top_k out of range, blank
        # text, a box without area.
        url = f"{ready.url}/v1/vision/segment"
        headers = {versions.HEADER: f"vision/{versions.VERSIONS['vision']}"}
        bad = [
            ({"text": "a", "points": [[1, 1]]}, 5),
            ({"text": "a"}, 0),
            ({"text": "  "}, 5),
            ({"boxes": [[10, 10, 10, 20]]}, 5),
        ]
        async with httpx.AsyncClient(trust_env=False) as http:
            for prompt, top_k in bad:
                sent = {"image": image.model_dump(mode="json"), "prompt": prompt, "top_k": top_k}
                response = await http.post(url, json=sent, headers=headers)
                assert response.status_code == 400, (prompt, response.text)
                assert response.json()["code"] == "invalid_request"


async def test_a_broken_model_stops_the_service_once_its_call_is_answered(
    tmp_path: Path,
) -> None:
    """A refused request leaves the service ready; a broken model answers its call (`not_ready`,
    an infrastructure error), takes the service out of service (refusing calls, nothing reaching
    the GPU thread) and stops it, so that its process exits to be restarted."""
    detector = ColorModel("colors")
    service = make_service(tmp_path, detector=detector)
    ready_file = tmp_path / "ready.json"
    stopped: list[ServiceFault] = []

    async def serve() -> None:
        try:
            await run(service, ready_file=ready_file)
        except ServiceFault as error:
            stopped.append(error)

    with anyio.fail_after(20):
        async with anyio.create_task_group() as tg, httpx.AsyncClient(trust_env=False) as http:
            tg.start_soon(serve)
            while (ready := read_ready(ready_file)) is None:  # noqa: ASYNC110
                await anyio.sleep(0.01)
            await wait_ready(http, ready.url)
            proxy = client(Vision, ready.url, token=None, timeout=10, timeouts=TIMEOUTS)
            image = await service.blobs.aput_array(scene())
            with pytest.raises(ProtocolError, match="99 tokens"):  # refused: it stays ready
                await proxy.detect(image, ["too long"])
            assert service.ready
            with pytest.raises(InfraError, match="RuntimeError on the GPU thread") as caught:
                await proxy.detect(image, ["explode"])
            assert caught.value.code == "not_ready"
            assert not service.ready and service.failure == "RuntimeError on the GPU thread"
            calls = detector.calls
            for call in (service.info, lambda: service.detect(image, ["red"])):
                with pytest.raises(InfraError, match=r"failed \(RuntimeError on the GPU thread\)"):
                    await call()
            assert detector.calls == calls  # nothing reached the GPU thread after the failure
            await aclose(proxy)
    expected = "the vision service failed: RuntimeError on the GPU thread"
    assert [str(error) for error in stopped] == [expected]
