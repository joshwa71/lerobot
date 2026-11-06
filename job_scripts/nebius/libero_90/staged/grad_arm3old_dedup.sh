#!/bin/bash
# E47 graduation: arm 3-OLD (this morning's DEDUPED warm-up, VLM knn36 — the compact
# "one state token" palette). Expert certified; VLM palette ~2 draws. Graduates as the
# compact-vs-broadcast fit comparison (its defects are family-side; the 5-task window
# has one basket task). NB its checkpoint lives on VM3 — rsync to the running box first.
WARM_RUN=libero_90_pi05_jointwarm10k_arm3_n256r2_vlmknn36
GRAD_TAG=arm3old_dedup_vlmknn36
source "$(dirname "$0")/joint_aphase_seq5_common.sh"
