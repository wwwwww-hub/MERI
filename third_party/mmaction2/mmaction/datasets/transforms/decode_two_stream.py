# Copyright (c) OpenMMLab.
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
from mmengine.registry import TRANSFORMS
from mmaction.datasets.transforms import RawFrameDecode


@TRANSFORMS.register_module()
class DecodeTwoStream:
    """
    解两路帧（RGB + Flow），与 RGB 对齐。

    - RGB 帧名：000000.jpg（6 位、从 0 起）
    - Flow 帧名：img_x0000.jpg / img_y0000.jpg（4 位、从 0 起）

    仅将 IO/解码相关参数传递给 RawFrameDecode（io_backend/decoding_backend/...），
    自定义的 root 参数仅用于拼接路径，不会传进 FileClient。
    参考：RawFrameDecode 需要的键与 FileClient 允许参数。
    """

    def __init__(
        self,
        # 根目录（可选；不传则使用 results['frame_dir']）
        rgb_root: Optional[str] = None,
        flow_root: Optional[str] = None,

        # —— 文件名模板 & 起始索引（按你的数据约定修改）——
        rgb_tmpl: str = "{:06d}.jpg",
        rgb_start_index: int = 0,
        flow_tmpl: str = "img_{}{:04d}.jpg",
        flow_start_index: int = 0,

        # RawFrameDecode 的 IO/解码参数
        io_backend: str = "disk",
        decoding_backend: str = "cv2",
        # 可选：memcached/path mapping 等
        server_list_cfg: Optional[str] = None,
        client_cfg: Optional[str] = None,
        sys_path: Optional[str] = None,
        path_mapping: Optional[dict] = None,
    ):
        # 仅用于拼路径
        self.rgb_root = rgb_root
        self.flow_root = flow_root

        # 模板与起始索引
        self.rgb_tmpl = rgb_tmpl
        self.rgb_start_index = rgb_start_index
        self.flow_tmpl = flow_tmpl
        self.flow_start_index = flow_start_index

        # 只构造 RawFrameDecode 支持的 kwargs
        decoder_kwargs: Dict[str, Any] = dict(
            io_backend=io_backend, decoding_backend=decoding_backend
        )
        if server_list_cfg is not None:
            decoder_kwargs["server_list_cfg"] = server_list_cfg
        if client_cfg is not None:
            decoder_kwargs["client_cfg"] = client_cfg
        if sys_path is not None:
            decoder_kwargs["sys_path"] = sys_path
        if path_mapping is not None:
            decoder_kwargs["path_mapping"] = path_mapping

        self._rgb_decoder = RawFrameDecode(**decoder_kwargs)
        self._flow_decoder = RawFrameDecode(**decoder_kwargs)

    @staticmethod
    def _as_np(x: Sequence[int] | np.ndarray) -> np.ndarray:
        return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=np.int64)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        # -------- RGB --------
        rgb_req = results.copy()
        rgb_req["modality"] = "RGB"
        rgb_req["filename_tmpl"] = self.rgb_tmpl
        rgb_req["start_index"] = self.rgb_start_index  # ← 000000.jpg 从 0 起

        if self.rgb_root is not None:
            rgb_req["frame_dir"] = os.path.join(
                self.rgb_root, os.path.basename(results["frame_dir"])
            )

        rgb_res = self._rgb_decoder(rgb_req)   # -> 'imgs'
        imgs_rgb = rgb_res.pop("imgs")         # (T,3,H,W)

        # -------- FLOW --------
        flow_req = results.copy()
        flow_req["modality"] = "Flow"
        flow_req["filename_tmpl"] = self.flow_tmpl
        flow_req["start_index"] = self.flow_start_index  # ← img_x0000 从 0 起

        flow_dir = results["frame_dir"]
        if self.flow_root is not None:
            flow_dir = os.path.join(self.flow_root, os.path.basename(results["frame_dir"]))
        flow_req["frame_dir"] = flow_dir

        # Flow 直接复用同一组 frame_inds（与 RGB 对齐）
        flow_req["frame_inds"] = self._as_np(results["frame_inds"])

        flow_res = self._flow_decoder(flow_req)
        imgs_flow = flow_res.pop("imgs")       # (T,2,H,W)

        # -------- 合并返回 --------
        out = rgb_res
        out["imgs_rgb"] = imgs_rgb
        out["imgs_flow"] = imgs_flow
        out["imgs"] = imgs_rgb  # 兼容旧管线
        return out
