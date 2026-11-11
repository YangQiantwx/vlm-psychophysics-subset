"""Shared T5 reference geometry and historical return-archive identity."""

from __future__ import annotations

from . import release as current_release


VERSION = "equal-canvas-v1"
MANIFEST = f"t5-{VERSION}/metadata.json"
SCOPES = ("classic/t5", "natural/t5")
LEGACY_SCOPES = tuple(f"{scope}/{VERSION}" for scope in current_release.PAIRED_SCOPES)
RELEASE_REVISION = "cdbda2c954c5ac70442e5dbaf8a649a163892d81"
MANIFEST_SHA256 = "635d50bb139e0ae5d05ddf429847bac18a592cc9af5916f33e7d395cf43259d3"
STATIC_PROMPT_SHA256 = (
    "292fe3e35590dc1dc6d90850fbf6dad43df773f68cf03051ed1bf73814e1355b"
)


def source_scope(scope):
    return scope.removesuffix(f"/{VERSION}")


def manifest_record():
    return {
        "path": MANIFEST,
        "sha256": MANIFEST_SHA256,
        "rows": 768,
        "revision": RELEASE_REVISION,
    }


def is_current_reference(scope):
    return scope in (*SCOPES, *LEGACY_SCOPES)
