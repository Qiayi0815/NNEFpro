set -e
module load python/3.10.12-fasrc01 >/dev/null 2>&1
source $(dirname $(dirname $(command -v python)))/etc/profile.d/conda.sh
E=/n/holylabs/LABS/atchan_lab/Lab/qzha/envs/pepbench
export PIP_CACHE_DIR=/n/holylabs/LABS/atchan_lab/Lab/qzha/pep_nmr_work/pipcache
conda create -y -p $E python=3.10 >/dev/null
conda activate $E
pip install -q torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q "transformers==4.48.1" accelerate pandas numpy scipy biopython gemmi openpyxl h5py pydssp requests
python -s -c "import torch,transformers,gemmi,pydssp;print(\"ENV_OK\",torch.__version__,transformers.__version__)"
