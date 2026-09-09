#!/bin/bash
# E68 C1 — zero-SEPARATION warm-up certificate on the paper cell (sep 8 -> 0; contrastive 0.05 kept).
# Pre-registration (E68): famIoU up from the 0.145 band; bgIoU up (the question); writable mass for
# the basket victims e0/e1 against e7's core down. Unchanged famIoU/bg => separation is decorative.
export ARM_TAG=e68c1_merged6x2_e468101416_v579111315_anchor040_sep0_c005_prepass
export SEP_W=0.0
export CONTRASTIVE_W=0.05
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/e68_c_cert_common.sh"
