"""Long-horizon identity memory and recovery modules."""

from identity_memory.identity_manager import (
    GlobalIdentityManager,
    GlobalIdentityRecord,
    IdentityState,
    RecoveryOutput,
    SingleCameraRecoveryPipeline,
)
from identity_memory.memory import IdentityMemory, SearchMatch
from identity_memory.multicamera_manager import (
    ConflictEvent,
    MultiCameraIdentityManager,
    MultiCameraIdentityRecord,
    MultiCameraRecoveryOutput,
)
from identity_memory.storage import (
    AuditLogEntry,
    IdentityStorage,
    LifecycleSnapshot,
    ObservationRecord,
    UnresolvedObservationRecord,
)
from identity_memory.tracklet_manager import (
    Tracklet,
    TrackletManager,
    TrackletSample,
    TrackletUpdateResult,
)

__all__ = [
    "AuditLogEntry",
    "ConflictEvent",
    "GlobalIdentityManager",
    "GlobalIdentityRecord",
    "IdentityMemory",
    "IdentityState",
    "IdentityStorage",
    "LifecycleSnapshot",
    "MultiCameraIdentityManager",
    "MultiCameraIdentityRecord",
    "MultiCameraRecoveryOutput",
    "ObservationRecord",
    "RecoveryOutput",
    "SearchMatch",
    "SingleCameraRecoveryPipeline",
    "Tracklet",
    "TrackletManager",
    "TrackletSample",
    "TrackletUpdateResult",
    "UnresolvedObservationRecord",
]
