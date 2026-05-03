from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from cart_stack.dashboard.runtime import DashboardController, RemoteRobotBackend
from cart_stack.shared.models import (
    ConnectionRequest,
    DriveCommand,
    MotorInvertChange,
    RlSessionStart,
    TerminalCommand,
)
from cart_stack.shared.physics import calculate_motion_model
from cart_stack.shared.models import MotionModelInput


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Autonomous Shopping Cart Dashboard")


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Force browsers to revalidate /static/ and / on every load so redeploys
    are picked up without a hard refresh."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheStaticMiddleware)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

controller = DashboardController()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def api_status() -> dict:
    return (await controller.status()).model_dump()


@app.post("/api/connection/connect")
async def api_connect(request: ConnectionRequest) -> dict:
    try:
        message, status = await controller.connect(request.target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": message, "status": status.model_dump()}


@app.post("/api/connection/disconnect")
async def api_disconnect() -> dict:
    message, status = await controller.disconnect()
    return {"message": message, "status": status.model_dump()}


@app.post("/api/drive")
async def api_drive(command: DriveCommand) -> dict:
    try:
        status = await controller.drive(command.left, command.right)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return status.model_dump()


@app.post("/api/stop")
async def api_stop() -> dict:
    try:
        status = await controller.stop()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return status.model_dump()


@app.post("/api/command")
async def api_command(command: TerminalCommand) -> dict:
    try:
        message, status = await controller.execute_terminal_command(command.command)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": message, "status": status.model_dump()}


@app.post("/api/physics")
async def api_physics(inputs: MotionModelInput) -> dict:
    return calculate_motion_model(inputs).model_dump()


@app.post("/api/motor-inverts")
async def api_motor_inverts(change: MotorInvertChange) -> dict:
    backend = controller._backend  # noqa: SLF001
    if not isinstance(backend, RemoteRobotBackend):
        raise HTTPException(status_code=400, detail="Connect to a Pi agent before changing motor directions.")
    try:
        await backend.set_motor_inverts(
            change.left1_invert,
            change.left2_invert,
            change.right1_invert,
            change.right2_invert,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await controller.status()).model_dump()


@app.post("/api/rl/start")
async def api_rl_start(request: RlSessionStart) -> dict:
    backend = controller._backend  # noqa: SLF001
    if not isinstance(backend, RemoteRobotBackend):
        raise HTTPException(status_code=400, detail="Connect to a Pi agent before starting an RL session.")
    try:
        await backend.rl_start(request.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await controller.status()).model_dump()


@app.post("/api/rl/stop")
async def api_rl_stop() -> dict:
    backend = controller._backend  # noqa: SLF001
    if not isinstance(backend, RemoteRobotBackend):
        raise HTTPException(status_code=400, detail="Connect to a Pi agent before stopping an RL session.")
    try:
        await backend.rl_stop()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return (await controller.status()).model_dump()


@app.get("/api/rl/log/list")
async def api_rl_log_list() -> dict:
    backend = controller._backend  # noqa: SLF001
    if not isinstance(backend, RemoteRobotBackend):
        raise HTTPException(status_code=400, detail="Connect to a Pi agent to list RL logs.")
    try:
        return await backend.rl_log_list()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/rl/log")
async def api_rl_log(name: str | None = None) -> StreamingResponse:
    backend = controller._backend  # noqa: SLF001
    if not isinstance(backend, RemoteRobotBackend):
        raise HTTPException(status_code=400, detail="Connect to a Pi agent to download RL logs.")

    # The remote stream yields a metadata dict first (contains filename),
    # then the raw file bytes. Pull the metadata before opening the response
    # so we can set the Content-Disposition header.
    iterator = backend.rl_log_stream(name)
    try:
        first = await iterator.__anext__()
    except StopAsyncIteration:
        raise HTTPException(status_code=404, detail="No RL log available.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = (first or {}).get("filename") or "rl_session.jsonl"

    async def stream_rest():
        async for chunk in iterator:
            if isinstance(chunk, dict):
                continue
            yield chunk

    return StreamingResponse(
        stream_rest(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/rl/policy/info")
async def api_rl_policy_info() -> dict:
    backend = controller._backend  # noqa: SLF001
    if not isinstance(backend, RemoteRobotBackend):
        raise HTTPException(status_code=400, detail="Connect to a Pi agent to read policy info.")
    try:
        return await backend.rl_policy_info()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/rl/policy/upload")
async def api_rl_policy_upload(request: Request) -> dict:
    backend = controller._backend  # noqa: SLF001
    if not isinstance(backend, RemoteRobotBackend):
        raise HTTPException(status_code=400, detail="Connect to a Pi agent before uploading a policy.")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty upload.")
    try:
        return await backend.rl_policy_upload(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


_BUILTIN_POLICY_DIR = BASE_DIR / "policies"


@app.get("/api/rl/policy/builtin/list")
async def api_rl_policy_builtin_list() -> dict:
    """List the policies bundled with the dashboard. They live in
    cart_stack/dashboard/policies/ and ship with the codebase."""
    if not _BUILTIN_POLICY_DIR.exists():
        return {"available": []}
    out = []
    for p in sorted(_BUILTIN_POLICY_DIR.glob("*.npz")):
        out.append({"name": p.stem, "size_bytes": p.stat().st_size})
    return {"available": out}


@app.post("/api/rl/policy/install-builtin")
async def api_rl_policy_install_builtin(name: str | None = None) -> dict:
    """Push one (or all) of the dashboard-bundled policies to the connected Pi.

    Query param `name`:
      - omitted / "all"  -> install every npz in the bundle
      - "bc_2d_combined_v2" / "bc_2d_5_1only_v2" -> install just that one
    """
    backend = controller._backend  # noqa: SLF001
    if not isinstance(backend, RemoteRobotBackend):
        raise HTTPException(status_code=400, detail="Connect to a Pi agent first.")
    if not _BUILTIN_POLICY_DIR.exists():
        raise HTTPException(status_code=500, detail="Dashboard has no bundled policies.")

    if name in (None, "all"):
        files = sorted(_BUILTIN_POLICY_DIR.glob("*.npz"))
    else:
        candidate = _BUILTIN_POLICY_DIR / f"{name}.npz"
        # prevent path traversal
        candidate = candidate.resolve()
        if not str(candidate).startswith(str(_BUILTIN_POLICY_DIR.resolve())):
            raise HTTPException(status_code=400, detail="Invalid policy name.")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"No bundled policy named {name}.")
        files = [candidate]

    results = []
    for fp in files:
        # The bundled file's stem (e.g. "bc_2d_combined_v2") is the target slot
        # name on the Pi. This lets us ship the two slot files
        # (bc_2d_combined_v2, bc_2d_5_1only_v2) and have each land in its own slot.
        slot_hint = fp.stem if fp.stem in (
            "bc_2d_combined_v2", "bc_2d_5_1only_v2",
            "bc_2d_combined_pre_fix", "bc_2d", "bc_2d_mirror",
            "bc_2d_awr",
        ) else None
        try:
            res = await backend.rl_policy_upload(fp.read_bytes(), slot=slot_hint)
            results.append({"file": fp.name, "ok": True, **res})
        except Exception as exc:
            results.append({"file": fp.name, "ok": False, "error": str(exc)})
    return {"installed": results}


@app.get("/api/camera/stream")
async def api_camera_stream() -> StreamingResponse:
    if not controller.can_stream_camera():
        raise HTTPException(status_code=400, detail="Connect to a remote Pi agent before requesting the proxied camera stream.")
    stream = controller.camera_stream()
    return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await controller.close()
