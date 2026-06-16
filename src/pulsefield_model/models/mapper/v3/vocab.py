from __future__ import annotations

from pulsefield_model.models.mapper.shared.vocab import (
    KEY_COUNT,
    LaneAction,
    MapperTupleVocab,
    coerce_lane_action,
)


ACTION_TO_SIGNATURE_SYMBOL = {
    LaneAction.NONE: ".",
    LaneAction.TAP: "T",
    LaneAction.HOLD_START: "S",
    LaneAction.HOLD_END: "E",
}
SIGNATURE_SYMBOL_TO_ACTION = {symbol: action for action, symbol in ACTION_TO_SIGNATURE_SYMBOL.items()}


class MapperV3Vocab(MapperTupleVocab):
    """Mapper v3 local event-group vocabulary.

    The token ids intentionally reuse the existing complete 4-lane EVENT
    vocabulary shape. The v3 contract is the representation semantics: one
    token describes the current same-time 4-lane event group, with no fallback
    stream and no cross-window references.
    """

    contract_name = "v3_event_groups"

    def event_signature(self, token_id: int) -> str:
        return event_signature(self.decode_event(int(token_id)))

    def event_token_id_from_signature(self, signature: str) -> int:
        return self.encode_event(actions_from_event_signature(signature))


def event_signature(lane_actions: tuple[LaneAction | str, ...]) -> str:
    actions = tuple(coerce_lane_action(action) for action in lane_actions)
    if len(actions) != KEY_COUNT:
        raise ValueError(f"event signature requires exactly {KEY_COUNT} actions: {actions}")
    signature = "".join(ACTION_TO_SIGNATURE_SYMBOL[action] for action in actions)
    if signature == "." * KEY_COUNT:
        raise ValueError("v3 EVENT token cannot represent an all-NONE event")
    return signature


def actions_from_event_signature(signature: str) -> tuple[LaneAction, LaneAction, LaneAction, LaneAction]:
    if len(signature) != KEY_COUNT:
        raise ValueError(f"event signature must contain {KEY_COUNT} symbols: {signature!r}")
    try:
        actions = tuple(SIGNATURE_SYMBOL_TO_ACTION[symbol] for symbol in signature)
    except KeyError as exc:
        raise ValueError(f"unsupported event signature symbol in {signature!r}") from exc
    if all(action == LaneAction.NONE for action in actions):
        raise ValueError("v3 EVENT token cannot represent an all-NONE event")
    return actions  # type: ignore[return-value]
