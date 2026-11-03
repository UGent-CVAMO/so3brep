# Shared by the reproduce scripts. Data root: $SO3BREP_DATA (default ./data). Extra arguments go to train.train.
set -euo pipefail
cd "$(dirname "$0")/.."
export SO3BREP_DATA="${SO3BREP_DATA:-data}"
train() { python -m train.train "$@"; }
