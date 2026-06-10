

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

current_time=$(date +"%Y-%m-%d-%H-%M-%S")
name=${1:-sample}
steps=${2:-200}

python "${SCRIPT_DIR}/main_navigo.py" \
    --input_data "${PROJECT_ROOT}/infer_reprogramming/data_store/adata_${name}.h5ad" \
    --output_dir "${PROJECT_ROOT}/results/${current_time}_${name}_${steps}" \
    --train_steps "${steps}" \
    --rounds 10 \
    --flow_steps 10 \
    --batch_size 25 \
    --learning_rate 4e-5
