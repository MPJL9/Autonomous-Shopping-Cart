from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cart_stack.dashboard.runtime import DashboardController
from cart_stack.shared.models import ConnectionRequest, DriveCommand, TerminalCommand
from cart_stack.shared.physics import calculate_motion_model
from cart_stack.shared.models import MotionModelInput


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Autonomous Shopping Cart Dashboard")
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


@app.get("/api/camera/stream")
async def api_camera_stream() -> StreamingResponse:
    if not controller.can_stream_camera():
        raise HTTPException(status_code=400, detail="Connect to a remote Pi agent before requesting the proxied camera stream.")
    stream = controller.camera_stream()
    return StreamingResponse(stream, media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await controller.close()
