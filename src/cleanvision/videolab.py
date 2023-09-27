"""Videolab is an extension of Imagelab for finding issues in a video dataset."""
import os
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import av  # type: ignore
from imagehash import phash
from PIL import Image

from cleanvision.imagelab import Imagelab

# custom type
SampleFrame = Tuple[Union[Image.Image, None], Dict[str, bool]]


@dataclass
class SamplerConfig:
    min_frame_interval_sec: float = 1
    keyframes_only: bool = True
    buffer_size: int = 10
    hash_size: int = 4
    queue_wait: float = 0.1
    debug: bool = False


class HashBuffer:
    def __init__(self, size: int) -> None:
        self.ordered_buffer: OrderedDict[
            str, Tuple[Image, Dict[str, Any]]
        ] = OrderedDict()
        self.max_size = size

    def add(
        self, item: Image, hash_: str, metadata: Dict[str, Any] = {}
    ) -> Optional[Any]:
        if not self.__check_duplicate(hash_):
            return self.__add(item, hash_, metadata)
        return None

    def __add(
        self, item: Image, hash_: str, metadata: Dict[str, Any] = {}
    ) -> Optional[Any]:
        self.ordered_buffer[hash_] = (item, metadata)
        if len(self.ordered_buffer) >= self.max_size:
            return self.ordered_buffer.popitem(last=False)[1]
        return None

    def __check_duplicate(self, hash_: str) -> bool:
        if hash_ in self.ordered_buffer:
            self.ordered_buffer.move_to_end(hash_)
            return True
        return False


class VideoSampler:
    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = cfg
        self.hash_buf = HashBuffer(cfg.buffer_size)
        self.stats: Counter[str] = Counter()

    def compute_hash(self, frame_img: Image) -> str:
        return str(phash(frame_img, hash_size=self.cfg.hash_size))

    def sample(self, video_path: str) -> Iterable[SampleFrame]:
        """Generate sample frames from a video"""
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if self.cfg.keyframes_only:
                stream.codec_context.skip_frame = "NONKEY"
            prev_time = -10
            for frame_indx, frame in enumerate(container.decode(stream)):
                # skip frames if keyframes_only is True
                time_diff = frame.time - prev_time
                self.stats["total"] += 1
                if time_diff < self.cfg.min_frame_interval_sec:
                    continue
                prev_time = frame.time

                frame_pil: Image = frame.to_image()
                frame_hash = self.compute_hash(frame_pil)

                res = self.hash_buf.add(
                    frame_pil,
                    frame_hash,
                    metadata={"frame_time": frame.time, "frame_indx": frame_indx},
                )
                self.stats["decoded"] += 1
                if res:
                    self.stats["produced"] += 1
                    yield res

        # flush buffer
        for _, item in self.hash_buf.ordered_buffer.items():
            if item:
                self.stats["produced"] += 1
                yield item
        yield None, {"end": True}

    def write_queue(self, video_path: str, q: Queue[SampleFrame]) -> None:
        for item in self.sample(video_path=video_path):
            q.put(item)


class Worker:
    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = cfg
        self.processor = VideoSampler(cfg=cfg)
        self.q: Queue[SampleFrame] = Queue()

    def launch(self, video_path: str, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        proc_thread = Thread(
            target=self.processor.write_queue, args=(video_path, self.q)
        )
        proc_thread.start()
        self.queue_reader(output_path, read_interval=self.cfg.queue_wait)
        proc_thread.join()

    def queue_reader(self, output_path: str, read_interval: float = 0.1) -> None:
        while True:
            if not self.q.empty():
                item = self.q.get()
                frame, metadata = item
                if frame is not None:
                    if isinstance(frame, Image.Image):
                        frame.save(
                            os.path.join(output_path, f"{metadata['frame_time']}.jpg")
                        )
                if metadata.get("end", False):
                    break
            time.sleep(read_interval)


class Videolab(Imagelab):
    pass
