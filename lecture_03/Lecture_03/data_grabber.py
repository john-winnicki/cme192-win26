# download_flowbench_ldc_128_to_mat.py
# pip install huggingface_hub numpy scipy

from huggingface_hub import hf_hub_download
import numpy as np
from scipy.io import savemat

REPO = "BGLab/FlowBench"
X_FILE = "LDC_NS_2D/128x128/nurbs_lid_driven_cavity_X.npz"
Y_FILE = "LDC_NS_2D/128x128/nurbs_lid_driven_cavity_Y.npz"

x_path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=X_FILE, local_dir="flowbench_data")
y_path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=Y_FILE, local_dir="flowbench_data")

X = np.load(x_path, allow_pickle=True)
Y = np.load(y_path, allow_pickle=True)

print("X keys:", X.files)
print("Y keys:", Y.files)
for k in X.files:
    arr = X[k]
    print("X", k, getattr(arr, "shape", None), type(arr))
for k in Y.files:
    arr = Y[k]
    print("Y", k, getattr(arr, "shape", None), type(arr))

mdict = {f"X_{k}": X[k] for k in X.files}
mdict.update({f"Y_{k}": Y[k] for k in Y.files})

savemat("flowbench_ldc_nurbs_128.mat", mdict, do_compression=True)
print("Wrote: flowbench_ldc_nurbs_128.mat")
