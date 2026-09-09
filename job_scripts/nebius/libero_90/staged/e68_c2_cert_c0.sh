#!/bin/bash
# E68 C2 — zero-CONTRASTIVE warm-up certificate on the paper cell (contrastive 0.05 -> 0; sep 8 kept).
# Pre-registration (E68): collapse before 1k (E45 signature, ~70% of reads on one slot) or sprawl
# (core50/effnum far outside the gate). A clean pass => contrastive is not load-bearing with pooled
# VLM routing. If the warm-up itself dies (NaN/collapse), that IS the certificate — record it.
export ARM_TAG=e68c2_merged6x2_e468101416_v579111315_anchor040_sep8_c0_prepass
export SEP_W=8.0
export CONTRASTIVE_W=0.0
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/e68_c_cert_common.sh"
