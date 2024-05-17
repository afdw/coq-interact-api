from __future__ import annotations
from typing import Any, Literal, Self, Type, cast
from collections.abc import Callable, Coroutine, Iterable
from pydantic import BaseModel, ConfigDict, model_validator, ValidationInfo, ValidatorFunctionWrapHandler, model_serializer, TypeAdapter
import json
import traceback
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
        if isinstance(obj, cls):
            return obj
        else:
            repr = ExternalRepr[A].model_validate(obj)
            return cls.model_construct(v=external_map[repr.external_id])


class Tactic[A]:
    pass


class LocalRequestBase[R: BaseModel](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    _result_type: Type[BaseModel] = Internal[None]

    def result_type(self) -> Type[R]:
        return cast(Type[R], self._result_type)

    @model_validator(mode="wrap")
    @classmethod
    def validate_model(cls, obj: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo) -> Self:
        obj_copy = {}
        for key, value in dict(obj).items():
            if isinstance(value, Internal):
                obj_copy[key] = dict(value)  # type: ignore
            else:
                obj_copy[key] = value
        return handler(obj_copy)


class LocalRequestUnit(LocalRequestBase[Internal[None]]):
    type: Literal["LocalRequestUnit"] = "LocalRequestUnit"
    _result_type: Type[BaseModel] = Internal[None]


class LocalRequestTacticReturn[A](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticReturn"] = "LocalRequestTacticReturn"
    _result_type: Type[BaseModel] = Internal[Tactic[object]]
    value: Internal[A]


class LocalRequestTacticBind[A, B](LocalRequestBase[Internal[Tactic[B]]]):
    type: Literal["LocalRequestTacticBind"] = "LocalRequestTacticBind"
    _result_type: Type[BaseModel] = Internal[Tactic[object]]
    tac: Internal[Tactic[A]]
    f: External[Callable[[Internal[A]], Coroutine[None, None, Internal[Tactic[B]]]]]


class LocalRequestTacticThen[A](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticThen"] = "LocalRequestTacticThen"
    _result_type: Type[BaseModel] = Internal[Tactic[object]]
    tac_1: Internal[Tactic[None]]
    tac_2: Internal[Tactic[A]]


class LocalRequestTacticOr[A](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticOr"] = "LocalRequestTacticOr"
    _result_type: Type[BaseModel] = Internal[Tactic[object]]
    tac_1: Internal[Tactic[A]]
    tac_2: Internal[Tactic[A]]


class LocalRequestTacticMessage(LocalRequestBase[Internal[Tactic[None]]]):
    type: Literal["LocalRequestTacticMessage"] = "LocalRequestTacticMessage"
    _result_type: Type[BaseModel] = Internal[Tactic[object]]
    msg: str


class LocalRequestTacticLtac(LocalRequestBase[Internal[Tactic[None]]]):
    type: Literal["LocalRequestTacticLtac"] = "LocalRequestTacticLtac"
    _result_type: Type[BaseModel] = Internal[Tactic[object]]
    tactic: str


type LocalRequest = (
    LocalRequestUnit
    | LocalRequestTacticReturn[object]
    | LocalRequestTacticBind[object, object]
    | LocalRequestTacticThen[object]
    | LocalRequestTacticOr[object]
    | LocalRequestTacticMessage
    | LocalRequestTacticLtac
)


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


class LocalException(Exception):
    pass


class Handler:
    async def handle_local_request[R: BaseModel](self, _local_request: LocalRequestBase[R]) -> R:
        raise NotImplementedError()

    async def unit(self) -> Internal[None]:
        return await self.handle_local_request(LocalRequestUnit.model_construct())

    async def tactic_return[A](self, value: Internal[A]) -> Internal[Tactic[A]]:
        return await self.handle_local_request(LocalRequestTacticReturn[A](value=value))

    async def tactic_fail[A](self, exn: Exception) -> Internal[Tactic[A]]:  # type: ignore
        async def k(_: Internal[None]) -> Internal[Tactic[A]]:
            raise exn

        return await self.tactic_bind(await self.tactic_return(await self.unit()), k)

    async def tactic_bind[A, B](self, tac: Internal[Tactic[A]], f: Callable[[Internal[A]], Coroutine[None, None, Internal[Tactic[B]]]]) -> Internal[Tactic[B]]:
        return await self.handle_local_request(
            LocalRequestTacticBind[A, B](
                tac=tac,
                f=External[Callable[[Internal[A]], Coroutine[None, None, Internal[Tactic[B]]]]].model_construct(v=f),
            )
        )

    async def tactic_then[A](self, tac_1: Internal[Tactic[None]], tac_2: Internal[Tactic[A]]) -> Internal[Tactic[A]]:
        return await self.handle_local_request(LocalRequestTacticThen(tac_1=tac_1, tac_2=tac_2))

    async def tactic_then_list(self, tacs: Iterable[Internal[Tactic[None]]]) -> Internal[Tactic[None]]:
        tac_result = await self.tactic_return(await self.unit())
        for tac in reversed(list(tacs)):
            tac_result = await self.tactic_then(tac, tac_result)
        return tac_result

    async def tactic_or[A](self, tac_1: Internal[Tactic[A]], tac_2: Internal[Tactic[A]]) -> Internal[Tactic[A]]:
        return await self.handle_local_request(LocalRequestTacticOr(tac_1=tac_1, tac_2=tac_2))

    async def tactic_or_list[A](self, tacs: Iterable[Internal[Tactic[A]]]) -> Internal[Tactic[A]]:
        tac_result: Internal[Tactic[A]] = await self.tactic_fail(RuntimeError("No tactic"))
        for tac in reversed(list(tacs)):
            tac_result = await self.tactic_or(tac, tac_result)
        return tac_result

    async def tactic_message(self, msg: str) -> Internal[Tactic[None]]:
        return await self.handle_local_request(LocalRequestTacticMessage(msg=msg))

    async def tactic_ltac(self, tactic: str) -> Internal[Tactic[None]]:
        return await self.handle_local_request(LocalRequestTacticLtac(tactic=tactic))


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
                        await self.handle_remote(t)
                        return await aux()
                    case _ if s.startswith("<- "):
                        t = s.removeprefix("<- ")
                        return local_request.result_type().model_validate_json(t)
                    case _ if s.startswith("!! "):
                        t = s.removeprefix("!! ")
                        raise LocalException(json.loads(t))
                    case _:
                        raise TypeError("Invalid message: " + s)

            return await aux()

        async def handle_remote(self, t: str) -> None:
            try:
                remote_request: RemoteRequest = TypeAdapter(RemoteRequest).validate_json(t)
                result = await self.handle_remote_request(remote_request)
                await websocket.send_text("<- " + result.model_dump_json())
            except Exception:
                await websocket.send_text("!! " + json.dumps(traceback.format_exc()))

    handler = HandlerImpl()

    while True:
        try:
            s = await websocket.receive_text()
        except WebSocketDisconnect:
            break
        assert s.startswith("-> ")
        t = s.removeprefix("-> ")
        await handler.handle_remote(t)

    await websocket.close()
