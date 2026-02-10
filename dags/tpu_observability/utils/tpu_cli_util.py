from enum import Enum


class TpuInfoCmd(Enum):
  HELP = "tpu-info -help"
  VERSION = "tpu-info --version"
  PROCESS = "tpu-info --process"


CLI_VALIDATION_SPEC: dict[TpuInfoCmd, list[str]] = {
    TpuInfoCmd.HELP: [
        "Display TPU info and metrics.",
        "options:",
        "-h, --help",
        "-v, --version",
        "-p, --process",
        "--streaming",
        "--rate RATE",
        "--list_metrics",
    ],
    TpuInfoCmd.VERSION: [
        "tpu-info version:",
        "libtpu version:",
        "accelerator type:",
    ],
    TpuInfoCmd.PROCESS: [
        "TPU Process Info",
        "Chip",
        "PID",
        "Process Name",
        "/dev/vfio/",
        "python",
    ],
}
