from __future__ import annotations
from typing import Any, Literal, Self, Type, cast
from collections.abc import Callable, Coroutine
from pydantic import BaseModel, ConfigDict, model_validator, ValidationInfo, ValidatorFunctionWrapHandler, model_serializer, TypeAdapter
from starlette.websockets import WebSocket, WebSocketDisconnect

type InternalId = int


class Internal[A](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    internal_id: InternalId


type ExternalId = int

external_id_ctr: ExternalId = 0


def new_external_id() -> ExternalId:
    global external_id_ctr
    external_id_ctr += 1
    return external_id_ctr


external_map: dict[ExternalId, Any] = {}


class ExternalRepr[A](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    external_id: ExternalId


class External[A](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    v: A

    @model_serializer(mode="plain")
    def serialize_model(self) -> Any:
        external_id = new_external_id()
        external_map[external_id] = self.v
        return ExternalRepr[A](external_id=external_id).model_dump()

    @model_validator(mode="wrap")
    @classmethod
    def validate_model(cls, obj: Any, _handler: ValidatorFunctionWrapHandler, _info: ValidationInfo) -> Self:
        repr = ExternalRepr[A].model_validate(obj)
        return cls.model_construct(v=external_map[repr.external_id])


class Tactic[A]:
    pass


class LocalRequestBase[R: BaseModel](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    _result_type: Type[BaseModel] = Internal[None]

    def result_type(self) -> Type[R]:
        return cast(Type[R], self._result_type)


class LocalRequestUnit(LocalRequestBase[Internal[None]]):
    type: Literal["LocalRequestUnit"] = "LocalRequestUnit"
    _result_type: Type[BaseModel] = Internal[None]


class LocalRequestTacticReturn[A](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticReturn"] = "LocalRequestTacticReturn"
    _result_type: Type[BaseModel] = Internal[Tactic[object]]
    value: Internal[A]


class LocalRequestTacticBind[A, B](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticBind"] = "LocalRequestTacticBind"
    _result_type: Type[BaseModel] = Internal[Tactic[object]]
    tac: Internal[Tactic[A]]
    f: External[Callable[[Internal[A]], Coroutine[None, None, Internal[Tactic[B]]]]]


type LocalRequest = LocalRequestUnit | LocalRequestTacticReturn[object] | LocalRequestTacticBind[object, object]


class RemoteRequestBase[R: BaseModel](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    _result_type: Type[BaseModel]


class RemoteRequestApplyFunction[A, B](RemoteRequestBase[Internal[B]]):
    type: Literal["RemoteRequestApplyFunction"] = "RemoteRequestApplyFunction"
    _result_type: Type[BaseModel] = Internal[object]
    f: External[Callable[[Internal[A]], Coroutine[None, None, Internal[Tactic[B]]]]]
    x: Internal[A]


class RemoteRequestGetTactic(RemoteRequestBase[Internal[Tactic[None]]]):
    type: Literal["RemoteRequestGetTactic"] = "RemoteRequestGetTactic"
    _result_type: Type[BaseModel] = Internal[Tactic[None]]


type RemoteRequest = RemoteRequestApplyFunction[object, object] | RemoteRequestGetTactic


class Handler:
    async def handle_local_request[R: BaseModel](self, _local_request: LocalRequestBase[R]) -> R:
        raise NotImplementedError()

    async def unit(self) -> Internal[None]:
        return await self.handle_local_request(LocalRequestUnit.model_construct())

    async def tactic_return[A](self, value: Internal[A]) -> Internal[Tactic[A]]:
        return await self.handle_local_request(LocalRequestTacticReturn[A].model_construct(value=value))

    async def tactic_bind[A, B](self, tac: Internal[Tactic[A]], f: Callable[[Internal[A]], Coroutine[None, None, Internal[Tactic[B]]]]) -> Internal[Tactic[A]]:
        return await self.handle_local_request(
            LocalRequestTacticBind[A, B].model_construct(
                tac=tac,
                f=External[Callable[[Internal[A]], Coroutine[None, None, Internal[Tactic[B]]]]].model_construct(v=f),
            )
        )


async def handle_websocket(websocket: WebSocket, get_tactic: Callable[[Handler], Coroutine[None, None, Internal[Tactic[None]]]]) -> None:
    class HandlerImpl(Handler):
        async def handle_remote_request[R: BaseModel](self, remote_request: RemoteRequestBase[R]) -> R:
            match remote_request:
                case RemoteRequestApplyFunction(f=f, x=x):  # type: ignore
                    return await f.v(x)  # type: ignore
                case RemoteRequestGetTactic():  # type: ignore
                    return await get_tactic(handler)  # type: ignore
                case _:
                    raise TypeError("Unknown remote request")

        async def handle_local_request[R: BaseModel](self, local_request: LocalRequestBase[R]) -> R:
            await websocket.send_text("-> " + local_request.model_dump_json())

            async def aux() -> R:
                s = await websocket.receive_text()
                match s:
                    case _ if s.startswith("-> "):
                        t = s.removeprefix("-> ")
                        remote_request: RemoteRequest = TypeAdapter(RemoteRequest).validate_json(t)
                        result = await self.handle_remote_request(remote_request)
                        await websocket.send_text("<- " + result.model_dump_json())
                        return await aux()
                    case _ if s.startswith("<- "):
                        t = s.removeprefix("<- ")
                        return local_request.result_type().model_validate_json(t)
                    case _:
                        raise TypeError("Invalid message: " + s)

            return await aux()

    handler = HandlerImpl()

    while True:
        try:
            s = await websocket.receive_text()
            assert s.startswith("-> ")
            t = s.removeprefix("-> ")
            remote_request: RemoteRequest = TypeAdapter(RemoteRequest).validate_json(t)
            result = await handler.handle_remote_request(remote_request)
            await websocket.send_text("<- " + result.model_dump_json())
        except WebSocketDisconnect:
            pass

    await websocket.close()
