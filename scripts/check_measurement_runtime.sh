#!/usr/bin/env bash
# Verify the platform before installing the pinned runtime. Usage: check_measurement_runtime.sh <runtime.json>
set -euo pipefail
"${REGISTRY_PYTHON:-python3}" - "$1" <<'PY'
import json
import re
import sys
from pathlib import Path
runtime = json.loads(Path(sys.argv[1]).read_text())
if not re.fullmatch(r'core@[^:]+:[a-f0-9]{12,40}', runtime['neat_core']['install_target']):
    sys.exit('measurement requires an immutable Core revision')
fields = dict(line.split('=', 1) for line in Path('/etc/buildinfo').read_text().splitlines() if '=' in line)
if fields.get('DISTRO_VERSION', '').strip().strip('"') != runtime['devkit']['platform_version']:
    sys.exit('DevKit platform differs from the planned runtime')
PY
