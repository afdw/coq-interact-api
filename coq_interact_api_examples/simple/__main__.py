from typing import Any
import argparse
from starlette.websockets import WebSocket
from fastapi import FastAPI
import uvicorn

import coq_interact_api
from coq_interact_api import Internal, Tactic, Handler

app = FastAPI()


@app.get("/")
async def get_root() -> Any:
    return {"Hello": "World"}


@app.websocket("/interact_empty")
async def websocket_interact_empty(websocket: WebSocket) -> None:
    await websocket.accept()

    async def get_tactic(handler: Handler) -> Internal[Tactic[None]]:
        """
        ```ocaml
        Proofview.tclUNIT ()
        ```
        """

        print("interact")
        return await handler.tactic_return(await handler.unit())

    await coq_interact_api.handle_websocket(websocket, get_tactic)


@app.websocket("/interact_bind")
async def websocket_interact_bind(websocket: WebSocket) -> None:
    await websocket.accept()

    async def get_tactic(handler: Handler) -> Internal[Tactic[None]]:
        """
        ```ocaml
        Proofview.tclUNIT () >>= fun _ ->
        Proofview.tclUNIT ()
        ```
        """

        async def k(_: Internal[None]):
            print("interact")
            return await handler.tactic_return(await handler.unit())

        return await handler.tactic_bind(await handler.tactic_return(await handler.unit()), k)

    await coq_interact_api.handle_websocket(websocket, get_tactic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the server.")
    parser.add_argument("-a", "--addr", type=str, default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.addr, port=args.port)
