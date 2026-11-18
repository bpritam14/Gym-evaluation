# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional MCP bridge to browser-use's native indexed DOM action registry."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from urllib.parse import urlsplit

from mcp import types
from mcp.server import Server
from mcp.server.fastmcp import Image
from mcp.server.stdio import stdio_server
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import async_playwright

from saas_bench.opencode.browser import BrowserTools, origin

# File operations remain OpenCode-native. No nested Agent/model loop, arbitrary
# JS, external search, or hidden extraction-model calls are exposed.
EXCLUDED = [
    "done",
    "search",
    "evaluate",
    "extract",
    "read_file",
    "write_file",
    "replace_file",
]
INDEXED = {
    "click",
    "input",
    "upload_file",
    "dropdown_options",
    "select_dropdown",
    "scroll",
}


class IndexedBrowserTools(BrowserTools):
    def __init__(self, config):
        super().__init__(config)
        if version("browser-use") != "0.13.7":
            raise RuntimeError("browser_use backend requires browser-use==0.13.7")
        from browser_use import BrowserSession
        from browser_use.tools.service import Tools

        self.current_observation = None
        self.native = Tools(exclude_actions=EXCLUDED)
        self.action_model = self.native.registry.create_action_model()
        from browser_use.filesystem.file_system import FileSystem

        # FileSystem initialization clears its own directory: always allocate a
        # fresh session root so an MCP restart never removes prior task files.
        self.file_system = FileSystem(
            mkdtemp(prefix="browser-use-", dir=self.files), create_default_files=False
        )
        self.session = BrowserSession(
            headless=True,
            chromium_sandbox=False,
            executable_path=config.get("executable_path"),
            user_data_dir=str(self.artifacts / "profile"),
            downloads_path=str(self.files / "downloads"),
            enable_default_extensions=False,
            highlight_elements=False,
            dom_highlight_elements=False,
            wait_between_actions=0.2,
            viewport=config.get("viewport", {"width": 1280, "height": 960}),
            args=["--host-resolver-rules=MAP localhost 127.0.0.1"],
        )

    async def start(self):
        # Chrome's Unix socket paths cannot exceed ~108 bytes. Keep the real
        # temporary storage under the caller's approved root, with only a short
        # ephemeral symlink locally. This MCP owns its own process/environment.
        self.socket_alias = TemporaryDirectory(prefix="saas-bu-", dir="/tmp")
        alias = Path(self.socket_alias.name) / "tmp"
        alias.symlink_to(
            Path(os.environ.get("TMPDIR", "/tmp")).resolve(), target_is_directory=True
        )
        previous_tmp = os.environ.get("TMPDIR")
        try:
            os.environ["TMPDIR"] = str(alias)
            await self.session.start()
        finally:
            if previous_tmp is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_tmp
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.connect_over_cdp(
            self.session.cdp_url
        )
        self.browser.on(
            "disconnected", lambda: self.record_browser_fault("browser_disconnected")
        )
        self.context = self.browser.contexts[0]

        # Guard before requests reach other app slots, including redirected and
        # popup documents. Playwright is used only for isolation/download capture.
        async def guard(route):
            request = route.request
            try:
                local = urlsplit(request.url).hostname in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
                if (request.is_navigation_request() or local) and origin(
                    request.url
                ) not in self.allowed:
                    await route.abort()
                    return
            except ValueError:
                await route.abort()
                return
            if request.is_navigation_request():
                try:
                    page = request.frame.page
                except PlaywrightError:
                    page = None
                if page in self.redirect_guards:
                    await self.redirect_guards[page]
            await route.continue_()

        await self.context.route("**/*", guard)
        self.context.on("page", self._new_page)
        for page in self.context.pages:
            self._new_page(page)
        if self.config.get("refresh_socket"):
            path = Path(self.config["refresh_socket"])
            if path.exists():
                raise RuntimeError("Browser refresh socket already exists")
            self.refresh_server = await asyncio.start_unix_server(
                self.refresh_connection, path=str(path), limit=4096
            )
            path.chmod(0o600)

    def _new_page(self, page):
        self.page = page
        page.on("crash", lambda: self.record_browser_fault("renderer_crash"))
        self.redirect_guards[page] = asyncio.create_task(self.guard_redirects(page))
        # browser-use owns downloads and dialogs; do not handle either twice.

    async def close(self):
        try:
            await super().close()
        finally:
            try:
                await self.session.kill()
            finally:
                if hasattr(self, "socket_alias"):
                    self.socket_alias.cleanup()

    def schemas(self):
        result = []
        for name, action in self.native.registry.registry.actions.items():
            schema = copy.deepcopy(action.param_model.model_json_schema())
            if name in INDEXED:
                schema.setdefault("properties", {})["observation_id"] = {
                    "type": "string",
                    "description": "ID of the latest indexed DOM observation.",
                }
                schema.setdefault("required", []).append("observation_id")
            result.append(
                types.Tool(
                    name=name, description=action.description, inputSchema=schema
                )
            )
        result.append(
            types.Tool(
                name="observe",
                description="Get fresh indexed DOM, tab IDs and screenshot; invalidates previous indices.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )
        )
        result.append(
            types.Tool(
                name="inspect_input",
                description="Preview a declared input or task working image/PDF page.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "page": {"type": "integer", "minimum": 1, "default": 1},
                    },
                    "required": ["filename"],
                    "additionalProperties": False,
                },
            )
        )
        return result

    async def observe(self, timeout=15000):
        self.current_observation = None
        self.sequence += 1
        state = await asyncio.wait_for(
            self.session.get_browser_state_summary(
                include_screenshot=True, cached=False
            ),
            timeout=max(20, timeout / 1000),
        )
        stem = f"{self.sequence:05d}"
        # browser-use intentionally omits screenshots for about:blank. The
        # OpenCode first-decision hook still requires a real current image.
        png = (
            base64.b64decode(state.screenshot)
            if state.screenshot
            else await self.session.take_screenshot(full_page=False)
        )
        payload = {
            "observation_id": stem,
            "url": state.url,
            "title": state.title,
            "snapshot": state.dom_state.llm_representation(),
            "tabs": [tab.model_dump(mode="json") for tab in state.tabs],
            "page_info": state.page_info.model_dump(mode="json")
            if state.page_info
            else None,
            "browser_errors": state.browser_errors,
            "state_error": state.state_error,
            "closed_popup_messages": state.closed_popup_messages,
            "viewport": self.config.get("viewport", {"width": 1280, "height": 960}),
            "screenshot_available": bool(png),
            "screenshot_source": "browser_use",
        }
        (self.artifacts / f"{stem}.json").write_text(json.dumps(payload))
        result = [json.dumps(payload)]
        if png:
            (self.artifacts / f"{stem}.png").write_bytes(png)
            result.append(Image(data=png, format="png"))
            self.current_observation = stem
        return result

    async def invoke(self, name, arguments):
        async with self.lock:
            if self.browser_fault:
                raise RuntimeError(
                    "Browser runtime failed; retry the episode in a fresh runtime"
                )
            args = dict(arguments)
            if name == "observe":
                return await self.observe()
            if name == "inspect_input":
                return self.preview(**args)
            if name not in self.native.registry.registry.actions:
                raise ValueError("Unknown or disabled browser action")
            if name in INDEXED:
                if (
                    self.current_observation is None
                    or args.pop("observation_id", None) != self.current_observation
                ):
                    raise ValueError(
                        "Stale observation_id: observe again before using an index"
                    )
            if name == "navigate":
                if (
                    origin(args["url"]) not in self.allowed
                    or urlsplit(args["url"]).path not in {"", "/"}
                    or urlsplit(args["url"]).query
                ):
                    raise ValueError(
                        "Navigate only to assigned app roots; use UI for internal navigation"
                    )
            available = [str(path) for path in self.inputs.values()]
            if name == "upload_file":
                try:
                    path = self.resolve_input(args["path"])
                except ValueError:
                    path = self.workspace_file(args["path"])
                if not path.is_file():
                    raise ValueError("Upload file does not exist")
                args["path"] = str(path)
                available.append(str(path))
            # Do not refresh the DOM between validating the snapshot and dispatch.
            action = self.action_model(**{name: args})
            self.current_observation = None  # consume before any possible mutation
            outcome = await self.native.act(
                action,
                self.session,
                available_file_paths=available,
                file_system=self.file_system,
            )
            result = [
                json.dumps(
                    {
                        "action": name,
                        "result": outcome.model_dump(mode="json", exclude_none=True),
                    }
                )
            ]
            # Never retry a possibly completed mutation because capture failed.
            try:
                result += await self.observe()
            except Exception as exc:
                self.sequence += 1  # invalidate the old map even when capture failed
                result.append(
                    json.dumps(
                        {
                            "screenshot_available": False,
                            "observation_error": str(exc)[:1000],
                            "instruction": "Observe to recover evidence; do not repeat a successful action.",
                        }
                    )
                )
            return result

    def preview(self, filename, page=1):
        try:
            path = self.resolve_input(filename)
        except ValueError:
            path = self.workspace_file(filename)
        if path.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(path) as document:
                if not 1 <= page <= len(document):
                    raise ValueError("PDF page out of range")
                png = (
                    document[page - 1]
                    .get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
                    .tobytes("png")
                )
                return [
                    json.dumps(
                        {"filename": filename, "page": page, "pages": len(document)}
                    ),
                    Image(data=png, format="png"),
                ]
        formats = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
        if path.suffix.lower() not in formats:
            raise ValueError("Only image/PDF previews are supported")
        return [
            filename,
            Image(data=path.read_bytes(), format=formats[path.suffix.lower()]),
        ]


async def main(config):
    tools = IndexedBrowserTools(config)
    server = Server("saas-browser-use")

    @server.list_tools()
    async def list_tools():
        return tools.schemas()

    @server.call_tool()
    async def call_tool(name, arguments):
        values = await tools.invoke(name, arguments or {})
        return [
            value.to_image_content()
            if isinstance(value, Image)
            else types.TextContent(type="text", text=value)
            for value in values
        ]

    try:
        await tools.start()
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        await tools.close()


if __name__ == "__main__":
    asyncio.run(main(json.loads(Path(sys.argv[1]).read_text())))
