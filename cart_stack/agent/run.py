from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "cart_stack.agent.app:app",
        host=os.getenv("CART_AGENT_HOST", "0.0.0.0"),
        port=int(os.getenv("CART_AGENT_PORT", "8001")),
        reload=os.getenv("CART_AGENT_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()

