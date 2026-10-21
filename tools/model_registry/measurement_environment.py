"""Record the actual board environment used for model measurements."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .errors import ValidationError

# The shared measurement runner fixes the clocks before this collector records their observed values.
MLA_FREQUENCY = Path("/sys/class/devfreq/5000000.mla/cur_freq")
CPU_FREQUENCY = Path("/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq")


def _require_internals_runtime(installed: dict[str, str]) -> None:
    """Require a versioned Internals runtime package in the dpkg inventory."""
    if any(name.partition(":")[0] == "neat-runtime" and version.strip() for name, version in installed.items()):
        return
    raise ValidationError("could not establish installed Neat Internals version")


def runner_environment() -> dict[str, Any]:
    """Record installation identity, without taking performance measurements."""
    buildinfo = Path("/etc/buildinfo").read_text(encoding="utf-8")
    packages = subprocess.run(
        ["dpkg-query", "-W", "-f=${db:Status-Abbrev}\t${binary:Package}\t${Version}\n", "*neat*"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    installed = {}
    for line in packages.splitlines():
        fields = line.split("\t", 2)
        if len(fields) == 3 and fields[0] == "ii " and fields[2].strip():
            installed[fields[1]] = fields[2]
    _require_internals_runtime(installed)
    return {
        "buildinfo": buildinfo,
        "packages": installed,
        "board_model": "Modalix",
        "machine": platform.machine(),
        "kernel_release": platform.release(),
        "cpu_count": os.cpu_count(),
        "cpu_governor": (
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
            if Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").exists()
            else None
        ),
        "mla_frequency_hz": (int(MLA_FREQUENCY.read_text(encoding="utf-8")) if MLA_FREQUENCY.exists() else None),
        "cpu_frequency_khz": (int(CPU_FREQUENCY.read_text(encoding="utf-8")) if CPU_FREQUENCY.exists() else None),
        "runner": os.environ.get("RUNNER_NAME", platform.node()),
    }
