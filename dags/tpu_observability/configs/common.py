from dataclasses import dataclass
from typing import Dict, Any

from dags.common.vm_resource import MachineVersion, TpuVersion


@dataclass(frozen=True)
class TpuConfig:
    tpu_type: str
    tpu_topology: str


# Only one version of the machine is supported at the moment.
# Other versions (e.g., "ct5p-hightpu-4t") may be introduced later.
MACHINE_CONFIG_MAP: Dict[MachineVersion, TpuConfig] = {
    MachineVersion.CT6E_STAND_4T: TpuConfig(
        tpu_type=f"v{TpuVersion.TRILLIUM.value}",
        tpu_topology="4x4",
    ),
}
