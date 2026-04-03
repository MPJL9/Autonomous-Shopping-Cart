from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "cart_stack.dashboard.app:app",
        host=os.getenv("CART_DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.getenv("CART_DASHBOARD_PORT", "8000")),
        reload=os.getenv("CART_DASHBOARD_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()

