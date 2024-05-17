from typing import Any
import argparse
from starlette.websockets import WebSocket
from fastapi import FastAPI
import uvicorn

import coq_interact_api
from coq_interact_api import Internal, TypeDescUnit, TypeDescList, Tactic, HypKindAssumption, HypKindDefinition, Hyp, Goal, Handler

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
        return await handler.tactic_return(None)

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

        async def k(_: None) -> Internal[Tactic[None]]:
            print("interact")
            return await handler.tactic_return(None)

        return await handler.tactic_bind(TypeDescUnit(), await handler.tactic_return(None), k)

    await coq_interact_api.handle_websocket(websocket, get_tactic)


@app.websocket("/interact_message")
async def websocket_interact_message(websocket: WebSocket) -> None:
    await websocket.accept()

    async def get_tactic(handler: Handler) -> Internal[Tactic[None]]:
        return await handler.tactic_message("interact")

    await coq_interact_api.handle_websocket(websocket, get_tactic)


@app.websocket("/interact_ltac")
async def websocket_interact_ltac(websocket: WebSocket) -> None:
    await websocket.accept()

    async def get_tactic(handler: Handler) -> Internal[Tactic[None]]:
        return await handler.tactic_ltac("auto")

    await coq_interact_api.handle_websocket(websocket, get_tactic)


@app.websocket("/interact_enter")
async def websocket_interact_enter(websocket: WebSocket) -> None:
    await websocket.accept()

    async def get_tactic(handler: Handler) -> Internal[Tactic[None]]:
        async def k(goal: Goal) -> Internal[Tactic[None]]:
            async def hyp_print(hyp: Hyp) -> str:
                match hyp.kind:
                    case HypKindAssumption():
                        return f"{hyp.name} : {await handler.constr_print(hyp.type_)}"
                    case HypKindDefinition(value=value):
                        return f"{hyp.name} : {await handler.constr_print(hyp.type_)} = {await handler.constr_print(value)}"

            async def goal_print(goal: Goal) -> str:
                return f"{", ".join([await hyp_print(hyp) for hyp in goal.hyps])} ⊢ {await handler.constr_print(goal.concl)}"

            return await handler.tactic_message(await goal_print(goal))

        return await handler.tactic_ignore(TypeDescList(element_type_desc=TypeDescUnit()), await handler.tactic_enter(k))

    await coq_interact_api.handle_websocket(websocket, get_tactic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the server.")
    parser.add_argument("-a", "--addr", type=str, default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.addr, port=args.port)
