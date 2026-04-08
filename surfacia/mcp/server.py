from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from .tooling import TOOL_SPECS, ToolSpec
from .utils import ToolExecutionError, serialize_jsonable

SERVER_NAME = "surfacia-mcp"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-03-26"


def _tool_map() -> dict[str, ToolSpec]:
    return {tool.name: tool for tool in TOOL_SPECS}


def _read_message(input_stream) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = input_stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8").strip()
        if not decoded:
            continue
        name, _, value = decoded.partition(":")
        headers[name.lower()] = value.strip()

    content_length = headers.get("content-length")
    if content_length is None:
        raise RuntimeError("Missing Content-Length header.")
    body = input_stream.read(int(content_length))
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(output_stream, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    output_stream.write(header)
    output_stream.write(body)
    output_stream.flush()


def _success_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": serialize_jsonable(result),
    }


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _initialize_result() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {
                "listChanged": False,
            }
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
    }


def _tools_list_result() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in TOOL_SPECS
        ]
    }


def _call_tool_result(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = _tool_map().get(name)
    if tool is None:
        raise ToolExecutionError(f"Unknown tool: {name}")

    structured = tool.handler(arguments or {})
    summary = structured.get("summary", f"Completed {name}.")
    return {
        "content": [
            {
                "type": "text",
                "text": summary,
            }
        ],
        "structuredContent": structured,
        "isError": not bool(structured.get("ok", True)),
    }


def _handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params", {})

    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _success_response(request_id, _initialize_result())
    if method == "ping":
        return _success_response(request_id, {})
    if method == "tools/list":
        return _success_response(request_id, _tools_list_result())
    if method == "tools/call":
        try:
            result = _call_tool_result(
                params.get("name", ""),
                params.get("arguments", {}) or {},
            )
            return _success_response(request_id, result)
        except ToolExecutionError as exc:
            return _success_response(
                request_id,
                {
                    "content": [{"type": "text", "text": str(exc)}],
                    "structuredContent": {
                        "ok": False,
                        "summary": str(exc),
                    },
                    "isError": True,
                },
            )
    return _error_response(request_id, -32601, f"Method not found: {method}")


def run_stdio_server() -> int:
    logger = logging.getLogger(SERVER_NAME)
    logger.info("Starting %s %s", SERVER_NAME, SERVER_VERSION)
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer

    while True:
        try:
            message = _read_message(input_stream)
            if message is None:
                break
            response = _handle_message(message)
            if response is not None:
                _write_message(output_stream, response)
        except Exception as exc:  # pragma: no cover - top-level safety
            logger.exception("Unhandled MCP server error")
            fallback_id = None
            if "message" in locals():
                fallback_id = message.get("id")
            response = _error_response(fallback_id, -32603, f"Internal error: {exc}")
            _write_message(output_stream, response)
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Surfacia MCP stdio server.")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level written to stderr.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())
