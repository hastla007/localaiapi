"""Async client utilities for interacting with a ComfyUI server.

This module provides a lightweight wrapper around the ComfyUI REST and
websocket interfaces. It is responsible for uploading assets, queuing
workflows, monitoring execution, and downloading the generated media
back to the API container.

The implementation favours resiliency: websocket progress updates are
used when available but the client gracefully falls back to periodic
polling if the socket connection cannot be established. Workflows are
loaded from ``/app/comfyui_workflows`` by default which allows the file
structure to be overridden via Docker volumes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import websockets
from PIL import Image

__all__ = ["ComfyUIClient", "get_comfyui_client"]

logger = logging.getLogger("comfyui")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class ComfyUIOutput:
    """Representation of a single ComfyUI output asset."""

    filename: str
    subfolder: str
    type: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComfyUIOutput":
        return cls(
            filename=data.get("filename", ""),
            subfolder=data.get("subfolder", ""),
            type=data.get("type", "output"),
        )


class ComfyUIClient:
    """Async helper for interacting with a ComfyUI server instance."""

    def __init__(self, base_url: Optional[str] = None, workflow_dir: Optional[str] = None):
        self.base_url = base_url or os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
        self.workflow_dir = Path(workflow_dir or os.getenv("COMFYUI_WORKFLOW_DIR", "/app/comfyui_workflows"))
        self._session: Optional[aiohttp.ClientSession] = None
        self._workflow_cache: Dict[str, Dict[str, Any]] = {}
        self._client_id = os.getenv("COMFYUI_CLIENT_ID", str(uuid.uuid4()))
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Session helpers
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Health checks
    async def health_check(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/system_stats") as response:
                if response.status != 200:
                    return False
                await response.json()
                return True
        except Exception as exc:  # pragma: no cover - network failures
            logger.debug("ComfyUI health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Workflow helpers
    def _load_workflow_template(self, workflow_name: str) -> Dict[str, Any]:
        if workflow_name not in self._workflow_cache:
            workflow_path = self.workflow_dir / workflow_name
            if not workflow_path.exists():
                raise FileNotFoundError(f"Workflow template not found: {workflow_path}")
            with workflow_path.open("r", encoding="utf-8") as handle:
                self._workflow_cache[workflow_name] = json.load(handle)
        # Return a deep copy to avoid mutating the cached template
        return json.loads(json.dumps(self._workflow_cache[workflow_name]))

    @staticmethod
    def _find_node_by_id(workflow: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        """Find node by its ID in API format workflow"""

        if node_id not in workflow:
            raise KeyError(f"Node ID '{node_id}' not found in workflow")
        return workflow[node_id]

    @staticmethod
    def _find_node_by_title(workflow: Dict[str, Any], title: str) -> tuple[str, Dict[str, Any]]:
        """Find node by title in API format workflow, returns (node_id, node)"""

        for node_id, node in workflow.items():
            if node.get("_meta", {}).get("title") == title:
                return node_id, node
        raise KeyError(f"Node with title '{title}' not found in workflow")

    @staticmethod
    def _find_node_by_class(workflow: Dict[str, Any], class_type: str) -> tuple[str, Dict[str, Any]]:
        """Find node by class_type in API format workflow, returns (node_id, node)"""

        for node_id, node in workflow.items():
            if node.get("class_type") == class_type:
                return node_id, node
        raise KeyError(f"Node with class_type '{class_type}' not found in workflow")

    def _set_node_input(self, workflow: Dict[str, Any], node_id: str, key: str, value: Any) -> None:
        """Set input value for a specific node by ID"""

        node = self._find_node_by_id(workflow, node_id)
        node.setdefault("inputs", {})[key] = value

    # ------------------------------------------------------------------
    # Upload helpers
    async def upload_image(self, image: Image.Image, filename: Optional[str] = None) -> str:
        """Upload an image to ComfyUI's input directory via the HTTP API."""

        filename = filename or f"input_{int(time.time())}.png"
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        form = aiohttp.FormData()
        form.add_field("image", buffer.getvalue(), filename=filename, content_type="image/png")

        session = await self._get_session()
        async with session.post(f"{self.base_url}/upload/image", data=form) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Failed to upload image to ComfyUI: {response.status} {text}")
            payload = await response.json()
            # ComfyUI returns {"name": "filename"}
            return payload.get("name", filename)

    # ------------------------------------------------------------------
    # Prompt execution helpers
    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        session = await self._get_session()
        payload = {"prompt": workflow, "client_id": self._client_id}
        async with session.post(f"{self.base_url}/prompt", json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Failed to queue ComfyUI workflow: {response.status} {text}")
            data = await response.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError("ComfyUI did not return a prompt_id")
            return prompt_id

    async def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
            if response.status != 200:
                return None
            payload = await response.json()
            return payload.get(prompt_id)

    async def _wait_ws(self, prompt_id: str, timeout: int) -> Dict[str, Any]:
        ws_url = self.base_url.replace("http", "ws", 1) + "/ws"
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20, max_size=None) as websocket:
            await websocket.send(json.dumps({"client_id": self._client_id}))
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                data = json.loads(message)
                event_type = data.get("type")
                payload = data.get("data", {})
                if event_type == "execution_error" and payload.get("prompt_id") == prompt_id:
                    raise RuntimeError(payload.get("error", "ComfyUI execution error"))
                if event_type == "execution_end" and payload.get("prompt_id") == prompt_id:
                    history = await self.get_history(prompt_id)
                    if history is None:
                        raise RuntimeError("ComfyUI finished but history is unavailable")
                    return history

    async def _wait_poll(self, prompt_id: str, timeout: int, interval: float = 2.0) -> Dict[str, Any]:
        start = time.time()
        while True:
            history = await self.get_history(prompt_id)
            if history:
                status = history.get("status", {}).get("status")
                if status == "completed":
                    return history
                if status == "error":
                    error = history.get("status", {}).get("error", "Unknown error")
                    raise RuntimeError(f"ComfyUI execution failed: {error}")
            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for ComfyUI result")
            await asyncio.sleep(interval)

    async def wait_for_completion(self, prompt_id: str, timeout: int = 600) -> Dict[str, Any]:
        try:
            return await self._wait_ws(prompt_id, timeout)
        except Exception as exc:
            logger.debug("Websocket wait failed (%s), falling back to polling", exc)
            return await self._wait_poll(prompt_id, timeout)

    # ------------------------------------------------------------------
    # Download helpers
    async def download_output(self, output: ComfyUIOutput, destination: Path) -> Path:
        params = {"filename": output.filename, "type": output.type}
        if output.subfolder:
            params["subfolder"] = output.subfolder

        session = await self._get_session()
        async with session.get(f"{self.base_url}/view", params=params) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Failed to download ComfyUI result: {response.status} {text}")
            data = await response.read()

        destination.mkdir(parents=True, exist_ok=True)
        result_path = destination / output.filename
        with result_path.open("wb") as handle:
            handle.write(data)
        return result_path

    # ------------------------------------------------------------------
    # High-level WAN 2.1 helper
    async def generate_video_wan21(
        self,
        *,
        image: Image.Image,
        prompt: Optional[str] = None,
        num_frames: int,
        steps: int,
        guidance_scale: float,
        fps: int,
        seed: Optional[int] = None,
    ) -> str:
        """Run the WAN 2.1 + LightX2V workflow and return the saved video path."""

        prompt = (prompt or "smooth camera movement, high quality").strip()

        # Resize image to match workflow expectations
        target_width = 1024 if image.width >= image.height else 576
        target_height = 576 if image.width >= image.height else 1024
        if image.width != target_width or image.height != target_height:
            image = image.resize((target_width, target_height))

        uploaded_name = await self.upload_image(image, filename="wan21_input.png")

        workflow = self._load_workflow_template("wan21_workflow.json")

        # Update node inputs based on your actual workflow structure

        # Node 52: LoadImage - set the uploaded image
        self._set_node_input(workflow, "52", "image", uploaded_name)

        # Node 6: Positive prompt
        self._set_node_input(workflow, "6", "text", prompt)

        # Node 7: Negative prompt (keep default or customize)
        # self._set_node_input(workflow, "7", "text", "your negative prompt")

        # Node 50: WanImageToVideo - main parameters
        self._set_node_input(workflow, "50", "length", num_frames)
        self._set_node_input(workflow, "50", "width", target_width)
        self._set_node_input(workflow, "50", "height", target_height)

        # Node 3: KSampler - sampling parameters
        self._set_node_input(workflow, "3", "steps", steps)
        self._set_node_input(workflow, "3", "cfg", guidance_scale)
        if seed is not None:
            self._set_node_input(workflow, "3", "seed", seed)

        # Node 30 & 331: Video output FPS
        self._set_node_input(workflow, "30", "frame_rate", fps)
        self._set_node_input(workflow, "331", "frame_rate", fps)

        prompt_id = await self.queue_prompt(workflow)
        history = await self.wait_for_completion(prompt_id)
        output = self._extract_first_video(history)
        destination = Path("/app/outputs")
        saved_path = await self.download_output(output, destination)
        return str(saved_path)

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_first_video(history: Dict[str, Any]) -> ComfyUIOutput:
        outputs = history.get("outputs", {})
        for node_outputs in outputs.values():
            video_items = node_outputs.get("video")
            if not video_items:
                continue
            return ComfyUIOutput.from_dict(video_items[0])
        raise RuntimeError("No video outputs returned by ComfyUI workflow")


_comfyui_client: Optional[ComfyUIClient] = None


def get_comfyui_client() -> ComfyUIClient:
    global _comfyui_client
    if _comfyui_client is None:
        _comfyui_client = ComfyUIClient()
    return _comfyui_client
