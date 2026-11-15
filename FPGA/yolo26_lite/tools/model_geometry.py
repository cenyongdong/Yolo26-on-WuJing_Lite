"""静态 YOLO26 one-to-one 接口尺寸。宽高按图像顺序传递，生成张量时按各自布局排列。"""
from dataclasses import dataclass
from math import prod


STRIDES = (8, 16, 32)
RAW_CHANNELS = 84


@dataclass(frozen=True)
class ModelGeometry:
    width: int = 640
    height: int = 640

    def __post_init__(self):
        if any(type(side) is not int or not 32 <= side <= 640 or side % max(STRIDES)
               for side in (self.width, self.height)):
            raise ValueError("Static input sides must be multiples of 32 in [32, 640]")

    @property
    def nhwc(self):
        return (1, self.height, self.width, 3)

    @property
    def nchw(self):
        return (1, 3, self.height, self.width)

    @property
    def anchors(self):
        return sum((self.width // stride) * (self.height // stride) for stride in STRIDES)

    @property
    def dense(self):
        return (1, RAW_CHANNELS, self.anchors)

    @property
    def rgb_bytes(self):
        return prod(self.nhwc)

    @classmethod
    def from_nhwc(cls, shape):
        if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
            raise ValueError("Expected batch-1 RGB NHWC input")
        return cls(shape[2], shape[1])


SQUARE = ModelGeometry()
