# shellcheck shell=bash
# Sourced (not run) by gaudi_check_cards.sh (idle baseline before launch) and gaudi_verify_busy.sh (busy check during training), so
# that both read the host's `hl-smi -Q index,module_id,memory.used -f csv` the same way.

# The ONE idle/busy threshold: a card whose memory.used is at most GAUDI_IDLE_MAX_MIB MiB is idle, above it is busy. Idle cards read
# 768 MiB (gaudi001/003/005); a training rank at batch 50 reads ~3219 MiB (job 64614665), at batch 200 ~42940 MiB (job 64613789).
# The earlier threshold of 4096 MiB sat above a healthy batch-50 rank and cancelled a good 4-card job; 1536 is twice the measured idle.
# shellcheck disable=SC2034  # read by the scripts that source this file
GAUDI_IDLE_MAX_MIB=1536

# gaudi_hlsmi_mem: stdin = that hl-smi output; prints one `index=MiB` line per row whose first two comma-separated fields are integers
# (so a header, a banner or a summary line is ignored); MiB is the third field without its "MiB" unit, or NA if that is not an integer
gaudi_hlsmi_mem() {
  awk -F, '{ gsub(/[[:space:]]/, "") } $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { m = $3; sub(/MiB$/, "", m); print $1 "=" ((m ~ /^[0-9]+$/) ? m + 0 : "NA") }'
}

# gaudi_alloc_mem <comma-separated allocated indices>: stdin = gaudi_hlsmi_mem output; prints one `index=MiB|NA|MISSING` line per
# allocated index, in the order given (MISSING: the index is not in the table)
gaudi_alloc_mem() {
  awk -F= -v want="$1" 'BEGIN { n = split(want, w, ",") } { mem[$1] = $2 } END { for (k = 1; k <= n; k++) print w[k] "=" ((w[k] in mem) ? mem[w[k]] : "MISSING") }'
}
