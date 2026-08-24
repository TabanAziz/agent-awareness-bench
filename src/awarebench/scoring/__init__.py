"""Scoring primitives: machine-checkable predicates over event logs."""

from awarebench.scoring.predicates import (
    REGISTRY,
    PredicateFactory,
    PredicateParams,
    constant_payload_field,
    event_type_absent_since,
    register,
)

__all__ = [
    "REGISTRY",
    "PredicateFactory",
    "PredicateParams",
    "constant_payload_field",
    "event_type_absent_since",
    "register",
]
