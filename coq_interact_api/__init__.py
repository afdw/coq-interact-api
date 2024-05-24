from __future__ import annotations
from typing import Any, Literal, Self, Type, cast, Annotated
from collections.abc import Callable, Coroutine, Iterable
from pydantic import (
    BaseModel,
    ConfigDict,
    WrapValidator,
    model_validator,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    model_serializer,
    TypeAdapter,
    Field,
    SkipValidation,
    SerializeAsAny,
)
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


class Constr:
    pass


class Tactic[A]:
    pass


class HypKindAssumption(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    type: Literal["Assumption"] = "Assumption"


class HypKindDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    type: Literal["Definition"] = "Definition"
    value: Internal[Constr]


type HypKind = Annotated[HypKindAssumption | HypKindDefinition, Field(discriminator="type")]


class Hyp(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    name: str
    type_: Internal[Constr]
    kind: HypKind


class Goal(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    hyps: list[Hyp]
    concl: Internal[Constr]

    def __hash__(self) -> int:
        return hash((tuple(self.hyps), self.concl))


def validate_type_desc(obj: Any, _handler: ValidatorFunctionWrapHandler, _info: ValidationInfo) -> TypeDescAux:
    return TypeAdapter(TypeDescAux).validate_python(obj)


class TypeDescBase[A](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)


class TypeDescUnit(TypeDescBase[None]):
    type: Literal["TypeDescUnit"] = "TypeDescUnit"


class TypeDescList[A](TypeDescBase[list[A]]):
    type: Literal["TypeDescList"] = "TypeDescList"
    element_type_desc: SerializeAsAny[Annotated[TypeDescBase[A], WrapValidator(validate_type_desc)]]


class TypeDescInternal[A](TypeDescBase[Internal[A]]):
    type: Literal["TypeDescInternal"] = "TypeDescInternal"


class TypeDescHyp(TypeDescBase[Hyp]):
    type: Literal["TypeDescHyp"] = "TypeDescHyp"


class TypeDescGoal(TypeDescBase[Goal]):
    type: Literal["TypeDescGoal"] = "TypeDescGoal"


type TypeDescAux = TypeDescUnit | TypeDescList[object] | TypeDescInternal[object] | TypeDescHyp | TypeDescGoal


type TypeDesc = Annotated[TypeDescAux, WrapValidator(validate_type_desc)]


def detect_type_desc[A](any: A) -> TypeDescBase[A]:
    if any is None:
        return TypeDescUnit()  # type: ignore
    if isinstance(any, list):
        return TypeDescList(element_type_desc=any[0] if any else TypeDescUnit())  # type: ignore
    if isinstance(any, Internal):
        return TypeDescInternal()
    if isinstance(any, Hyp):
        return TypeDescHyp()  # type: ignore
    if isinstance(any, Goal):
        return TypeDescGoal()  # type: ignore
    raise TypeError("Can not determine type desc")


def type_desc_to_type[A](type_desc: TypeDescBase[A]) -> Type[A]:
    match type_desc:
        case TypeDescUnit():  # type: ignore
            return None  # type: ignore
        case TypeDescList(element_type_desc=element_type_desc):  # type: ignore
            return list[type_desc_to_type(element_type_desc)]  # type: ignore
        case TypeDescInternal():  # type: ignore
            return Internal  # type: ignore
        case TypeDescHyp():  # type: ignore
            return Hyp  # type: ignore
        case TypeDescGoal():  # type: ignore
            return Goal  # type: ignore
        case _:
            raise TypeError("Invalid type desc: " + repr(type_desc))


class LocalRequestBase[R](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    _result_type: Type[object] = Internal[None]

    def result_type(self) -> Type[R]:
        return cast(Type[R], self._result_type)


class LocalRequestConstrPrint(LocalRequestBase[str]):
    type: Literal["LocalRequestConstrPrint"] = "LocalRequestConstrPrint"
    _result_type: Type[object] = str
    constr: Annotated[Internal[Constr], SkipValidation]


class LocalRequestTacticReturn[A](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticReturn"] = "LocalRequestTacticReturn"
    _result_type: Type[object] = Internal[Tactic[object]]
    type_desc: Annotated[SerializeAsAny[TypeDescBase[A]], SkipValidation]
    value: Annotated[A, SkipValidation]


class LocalRequestTacticBind[A, B](LocalRequestBase[Internal[Tactic[B]]]):
    type: Literal["LocalRequestTacticBind"] = "LocalRequestTacticBind"
    _result_type: Type[object] = Internal[Tactic[object]]
    type_desc: Annotated[SerializeAsAny[TypeDescBase[A]], SkipValidation]
    tac: Annotated[Internal[Tactic[A]], SkipValidation]
    f: Annotated[External[Callable[[A], Coroutine[None, None, Internal[Tactic[B]]]]], SkipValidation]


class LocalRequestTacticThen[A](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticThen"] = "LocalRequestTacticThen"
    _result_type: Type[object] = Internal[Tactic[object]]
    tac_1: Annotated[Internal[Tactic[None]], SkipValidation]
    tac_2: Annotated[Internal[Tactic[A]], SkipValidation]


class LocalRequestTacticOr[A](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticOr"] = "LocalRequestTacticOr"
    _result_type: Type[object] = Internal[Tactic[object]]
    tac_1: Annotated[Internal[Tactic[A]], SkipValidation]
    tac_2: Annotated[Internal[Tactic[A]], SkipValidation]


class LocalRequestTacticComplete[A](LocalRequestBase[Internal[Tactic[A]]]):
    type: Literal["LocalRequestTacticComplete"] = "LocalRequestTacticComplete"
    _result_type: Type[object] = Internal[Tactic[object]]
    tac: Annotated[Internal[Tactic[A]], SkipValidation]


class LocalRequestTacticGoals(LocalRequestBase[Internal[Tactic[list[Goal]]]]):
    type: Literal["LocalRequestTacticGoals"] = "LocalRequestTacticGoals"
    _result_type: Type[object] = Internal[Tactic[Internal[list[Goal]]]]


class LocalRequestTacticDispatch[A](LocalRequestBase[Internal[Tactic[list[A]]]]):
    type: Literal["LocalRequestTacticDispatch"] = "LocalRequestTacticDispatch"
    _result_type: Type[object] = Internal[Tactic[Internal[list[object]]]]
    tacs: Annotated[list[Internal[Tactic[A]]], SkipValidation]


class LocalRequestTacticMessage(LocalRequestBase[Internal[Tactic[None]]]):
    type: Literal["LocalRequestTacticMessage"] = "LocalRequestTacticMessage"
    _result_type: Type[object] = Internal[Tactic[object]]
    msg: str


class LocalRequestTacticLtac(LocalRequestBase[Internal[Tactic[None]]]):
    type: Literal["LocalRequestTacticLtac"] = "LocalRequestTacticLtac"
    _result_type: Type[object] = Internal[Tactic[object]]
    tactic: str


type LocalRequest = (
    LocalRequestConstrPrint
    | LocalRequestTacticReturn[object]
    | LocalRequestTacticBind[object, object]
    | LocalRequestTacticThen[object]
    | LocalRequestTacticOr[object]
    | LocalRequestTacticComplete[object]
    | LocalRequestTacticGoals
    | LocalRequestTacticDispatch[object]
    | LocalRequestTacticMessage
    | LocalRequestTacticLtac
)


class RemoteRequestBase[R](BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    _result_type: Type[object]


class RemoteRequestApplyFunction[A, B](RemoteRequestBase[Internal[B]]):
    type: Literal["RemoteRequestApplyFunction"] = "RemoteRequestApplyFunction"
    _result_type: Type[object] = Internal[object]
    type_desc: TypeDescBase[A]
    f: External[Callable[[A], Coroutine[None, None, Internal[Tactic[B]]]]]
    x: A

    @model_validator(mode="wrap")
    @classmethod
    def validate_model(cls, obj: Any, handler: ValidatorFunctionWrapHandler, _info: ValidationInfo) -> Self:
        if obj["type"] == "RemoteRequestApplyFunction":
            type_desc = TypeAdapter(TypeDesc).validate_python(obj["type_desc"])
            f = External[Callable[[A], Coroutine[None, None, Internal[Tactic[B]]]]].model_validate(obj["f"])
            x = TypeAdapter(type_desc_to_type(type_desc)).validate_python(obj["x"])
            return cls.model_construct(type_desc=type_desc, f=f, x=x)
        else:
            return handler(obj)


class RemoteRequestGetTactic(RemoteRequestBase[Internal[Tactic[None]]]):
    type: Literal["RemoteRequestGetTactic"] = "RemoteRequestGetTactic"
    _result_type: Type[object] = Internal[Tactic[None]]


type RemoteRequest = RemoteRequestApplyFunction[object, object] | RemoteRequestGetTactic


class LocalException(Exception):
    pass


class Handler:
    async def handle_local_request[R](self, _local_request: LocalRequestBase[R]) -> R:
        raise NotImplementedError()

    async def constr_print(self, constr: Internal[Constr]) -> str:
        return await self.handle_local_request(LocalRequestConstrPrint(constr=constr))

    async def tactic_return[A](self, value: A, type_desc: TypeDescBase[A] | None = None) -> Internal[Tactic[A]]:
        if type_desc is None:
            type_desc = detect_type_desc(value)
        return await self.handle_local_request(LocalRequestTacticReturn[A](type_desc=type_desc, value=value))

    async def tactic_fail[A](self, exn: Exception) -> Internal[Tactic[A]]:  # type: ignore
        async def k(_: None) -> Internal[Tactic[A]]:
            raise exn

        return await self.tactic_bind(TypeDescUnit(), await self.tactic_return(None), k)

    async def tactic_bind[A, B](
        self,
        type_desc: TypeDescBase[A],
        tac: Internal[Tactic[A]],
        f: Callable[[A], Coroutine[None, None, Internal[Tactic[B]]]],
    ) -> Internal[Tactic[B]]:
        return await self.handle_local_request(
            LocalRequestTacticBind[A, B](
                type_desc=type_desc,
                tac=tac,
                f=External[Callable[[A], Coroutine[None, None, Internal[Tactic[B]]]]].model_construct(v=f),
            )
        )

    async def tactic_ignore[A](self, type_desc: TypeDescBase[A], tac: Internal[Tactic[A]]) -> Internal[Tactic[None]]:
        async def k(_: A) -> Internal[Tactic[None]]:
            return await self.tactic_return(None)

        return await self.tactic_bind(type_desc, tac, k)

    async def tactic_then[A](self, tac_1: Internal[Tactic[None]], tac_2: Internal[Tactic[A]]) -> Internal[Tactic[A]]:
        return await self.handle_local_request(LocalRequestTacticThen(tac_1=tac_1, tac_2=tac_2))

    async def tactic_then_list(self, tacs: Iterable[Internal[Tactic[None]]]) -> Internal[Tactic[None]]:
        tac_result = await self.tactic_return(None)
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

    async def tactic_complete[A](self, tac: Internal[Tactic[A]]) -> Internal[Tactic[A]]:
        return await self.handle_local_request(LocalRequestTacticComplete(tac=tac))

    async def tactic_goals(self) -> Internal[Tactic[list[Goal]]]:
        return await self.handle_local_request(LocalRequestTacticGoals())

    async def tactic_dispatch[A](self, tacs: list[Internal[Tactic[A]]]) -> Internal[Tactic[list[A]]]:
        return await self.handle_local_request(LocalRequestTacticDispatch[A](tacs=tacs))

    async def tactic_enter[A](self, tac: Callable[[Goal], Coroutine[None, None, Internal[Tactic[A]]]]) -> Internal[Tactic[list[A]]]:
        async def k(goals: list[Goal]) -> Internal[Tactic[list[A]]]:
            return await self.tactic_dispatch([await tac(goal) for goal in goals])

        return await self.tactic_bind(TypeDescList(element_type_desc=TypeDescGoal()), await self.tactic_goals(), k)

    async def tactic_message(self, msg: str) -> Internal[Tactic[None]]:
        return await self.handle_local_request(LocalRequestTacticMessage(msg=msg))

    async def tactic_ltac(self, tactic: str) -> Internal[Tactic[None]]:
        return await self.handle_local_request(LocalRequestTacticLtac(tactic=tactic))


async def handle_websocket(websocket: WebSocket, get_tactic: Callable[[Handler], Coroutine[None, None, Internal[Tactic[None]]]]) -> None:
    class HandlerImpl(Handler):
        async def handle_remote_request[R](self, remote_request: RemoteRequestBase[R]) -> R:
            match remote_request:
                case RemoteRequestApplyFunction(f=f, x=x):  # type: ignore
                    return await f.v(x)  # type: ignore
                case RemoteRequestGetTactic():  # type: ignore
                    return await get_tactic(handler)  # type: ignore
                case _:
                    raise TypeError("Unknown remote request")

        async def handle_local_request[R](self, local_request: LocalRequestBase[R]) -> R:
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
                        return TypeAdapter(local_request.result_type()).validate_json(t)
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
