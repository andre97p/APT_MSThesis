#!/usr/bin/env bash
#
# snort_ids.sh — bring the Snort IDS/IPS up (and configure the capture path)
# for the PenGym cyber-range training platform.
#
# Snort runs INLINE via NFQUEUE (`--daq nfq --daq-var queue=N -Q`), so it only
# sees packets that netfilter hands to that queue. This script installs the
# steering rules + bridge-netfilter settings that feed the queue, makes sure the
# local ruleset is loadable, starts the service, and verifies capture is live.
#
# Usage:
#   ./snort_ids.sh          # or `up`  : configure + start (default)
#   ./snort_ids.sh down     #            remove steering rules + stop Snort
#   ./snort_ids.sh status   #            show current state
#
# Run it once before launching training (e.g. before `python run.py`).
# It is idempotent: safe to run repeatedly.

set -euo pipefail
export PATH=/usr/sbin:/sbin:/usr/bin:/bin:${PATH:-}

# ----------------------------------------------------------------------------
# Configuration (override via environment if your range differs)
# ----------------------------------------------------------------------------
RANGE_ID="${RANGE_ID:-126}"                       # cyber-range id -> bridges br<ID>-*
RANGE_CIDR="${RANGE_CIDR:-${RANGE_ID}.1.0.0/16}"  # fallback supernet if no bridges found
QUEUE_NUM="${QUEUE_NUM:-1}"                        # MUST match the service's --daq-var queue=
SNORT_SERVICE="${SNORT_SERVICE:-snort}"
SNORT_CONF="${SNORT_CONF:-/etc/snort/snort.conf}"
LOCAL_RULES="${LOCAL_RULES:-/etc/snort/rules/local.rules}"
RULES_BASELINE="${RULES_BASELINE:-/etc/snort/rules/local.rules.bak}"  # used by unlock_connections()
FAST_LOG="${FAST_LOG:-/var/log/snort/snort.alert.fast}"
# Fail-open: if Snort is not listening on the queue, ACCEPT packets instead of
# dropping them (a crashed IDS won't blackhole the whole range). Set to "" for
# strict fail-closed IPS behaviour.
QUEUE_BYPASS="${QUEUE_BYPASS:---queue-bypass}"

log()  { printf '\033[1;34m[snort-ids]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[snort-ids] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[snort-ids] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# Re-exec as root if needed (iptables / modprobe / systemctl require it)
# ----------------------------------------------------------------------------
if [[ ${EUID} -ne 0 ]]; then
    exec sudo -E bash "$0" "$@"
fi

# ----------------------------------------------------------------------------
# Discover the range bridges to monitor
# ----------------------------------------------------------------------------
discover_bridges() {
    if [[ -n "${SNORT_RANGE_BRIDGES:-}" ]]; then
        echo "${SNORT_RANGE_BRIDGES}"; return
    fi
    # All bridge interfaces whose name starts with br<RANGE_ID>- (e.g. br126-1-3)
    ip -br link show type bridge 2>/dev/null \
        | awk -v p="br${RANGE_ID}-" '$1 ~ "^"p {print $1}' | tr '\n' ' '
}

# ----------------------------------------------------------------------------
# Steering-rule helpers (idempotent: check-then-add / delete-if-present)
# ----------------------------------------------------------------------------
rule_specs() {
    # Emit the FORWARD rule bodies (without the -A/-D/-C verb) for the current range.
    local bridges; bridges="$(discover_bridges)"
    if [[ -n "${bridges// }" ]]; then
        for br in ${bridges}; do
            echo "-i ${br} -j NFQUEUE --queue-num ${QUEUE_NUM} ${QUEUE_BYPASS}"
            echo "-o ${br} -j NFQUEUE --queue-num ${QUEUE_NUM} ${QUEUE_BYPASS}"
        done
    else
        warn "no bridges matching br${RANGE_ID}-* found; falling back to supernet ${RANGE_CIDR}"
        echo "-s ${RANGE_CIDR} -j NFQUEUE --queue-num ${QUEUE_NUM} ${QUEUE_BYPASS}"
        echo "-d ${RANGE_CIDR} -j NFQUEUE --queue-num ${QUEUE_NUM} ${QUEUE_BYPASS}"
    fi
}

install_steering() {
    local spec
    while IFS= read -r spec; do
        [[ -z "${spec}" ]] && continue
        # shellcheck disable=SC2086
        if iptables -C FORWARD ${spec} 2>/dev/null; then
            log "steering rule already present: FORWARD ${spec}"
        else
            # shellcheck disable=SC2086
            iptables -A FORWARD ${spec}
            log "installed steering rule: FORWARD ${spec}"
        fi
    done < <(rule_specs)
}

remove_steering() {
    local spec
    while IFS= read -r spec; do
        [[ -z "${spec}" ]] && continue
        # shellcheck disable=SC2086
        while iptables -C FORWARD ${spec} 2>/dev/null; do
            # shellcheck disable=SC2086
            iptables -D FORWARD ${spec}
            log "removed steering rule: FORWARD ${spec}"
        done
    done < <(rule_specs)
}

# ----------------------------------------------------------------------------
# Bridge netfilter: let SAME-subnet (L2-switched) frames reach iptables/NFQUEUE
# ----------------------------------------------------------------------------
enable_bridge_netfilter() {
    modprobe br_netfilter 2>/dev/null || warn "could not load br_netfilter"
    sysctl -qw net.bridge.bridge-nf-call-iptables=1  2>/dev/null || warn "bridge-nf-call-iptables not settable"
    sysctl -qw net.bridge.bridge-nf-call-ip6tables=1 2>/dev/null || true
    # Persist across reboot (idempotent).
    echo br_netfilter > /etc/modules-load.d/br_netfilter.conf
    printf 'net.bridge.bridge-nf-call-iptables=1\nnet.bridge.bridge-nf-call-ip6tables=1\n' \
        > /etc/sysctl.d/99-snort-range.conf
    log "bridge-netfilter enabled (intra-subnet traffic is now visible to Snort)"
}

# ----------------------------------------------------------------------------
# Make the local ruleset loadable: repair Word-style smart quotes in msg:"..."
# ----------------------------------------------------------------------------
# Repair Word-style smart quotes (U+201C/U+201D) -> ASCII " in a single rules file.
_repair_quotes() {
    local f="$1"
    [[ -f "${f}" ]] || return 0
    if LC_ALL=C grep -qaF -e $'\xe2\x80\x9c' -e $'\xe2\x80\x9d' "${f}"; then
        cp -a "${f}" "${f}.orig-$(date +%Y%m%d%H%M%S)"
        LC_ALL=C sed -i 's/\xe2\x80\x9c/"/g; s/\xe2\x80\x9d/"/g' "${f}"
        log "repaired curly quotes in ${f} (backup kept)"
    fi
}

sanitize_rules() {
    [[ -f "${LOCAL_RULES}" ]] || { warn "${LOCAL_RULES} not found; skipping rule sanitize"; return; }
    _repair_quotes "${LOCAL_RULES}"
    if [[ -f "${RULES_BASELINE}" ]]; then
        # unlock_connections() restores this baseline on EVERY episode reset, so a
        # broken rule here would silently come back mid-training — keep it clean too.
        _repair_quotes "${RULES_BASELINE}"
    else
        cp -a "${LOCAL_RULES}" "${RULES_BASELINE}"
        log "seeded clean ruleset baseline at ${RULES_BASELINE}"
    fi
}

# ----------------------------------------------------------------------------
# Service control + verification
# ----------------------------------------------------------------------------
start_snort() {
    if systemctl is-active --quiet "${SNORT_SERVICE}"; then
        if systemctl reload "${SNORT_SERVICE}" 2>/dev/null; then
            log "reloaded ${SNORT_SERVICE}"
        else
            warn "reload failed — restarting ${SNORT_SERVICE}"
            systemctl restart "${SNORT_SERVICE}"
        fi
    else
        systemctl start "${SNORT_SERVICE}"
        log "started ${SNORT_SERVICE}"
    fi
    # Give it a moment to bind the queue / parse rules.
    for _ in $(seq 1 10); do
        systemctl is-active --quiet "${SNORT_SERVICE}" && break
        sleep 0.5
    done
    systemctl is-active --quiet "${SNORT_SERVICE}" \
        || die "${SNORT_SERVICE} failed to start — check: journalctl -u ${SNORT_SERVICE} -n40"
}

verify() {
    local ok=1
    systemctl is-active --quiet "${SNORT_SERVICE}" \
        && log "service ${SNORT_SERVICE}: active" \
        || { warn "service ${SNORT_SERVICE}: NOT active"; ok=0; }

    # Is a process bound to our queue? Column 2 (peer_portid) must be non-zero.
    local qline; qline="$(awk -v q="${QUEUE_NUM}" '$1==q {print}' /proc/net/netfilter/nfnetlink_queue 2>/dev/null || true)"
    if [[ -n "${qline}" ]] && [[ "$(echo "${qline}" | awk '{print $2}')" -gt 0 ]]; then
        log "NFQUEUE ${QUEUE_NUM}: bound (portid $(echo "${qline}" | awk '{print $2}'), queued=$(echo "${qline}" | awk '{print $3}'))"
    else
        warn "NFQUEUE ${QUEUE_NUM}: no process bound — Snort is not reading the queue"; ok=0
    fi

    # Any parse errors on the last (re)load?
    if journalctl -u "${SNORT_SERVICE}" -b -n 200 --no-pager 2>/dev/null | grep -qiE 'FATAL|Fatal Error'; then
        warn "recent FATAL in ${SNORT_SERVICE} journal — inspect: journalctl -u ${SNORT_SERVICE} -n40"; ok=0
    fi

    log "steering rules in FORWARD:"; iptables -vnL FORWARD --line-numbers | grep -i NFQUEUE || warn "  (none matched NFQUEUE)"
    log "alerts land in: ${FAST_LOG}  (watch live with: sudo tail -f ${FAST_LOG})"
    [[ ${ok} -eq 1 ]] && log "capture path OK ✔" || warn "capture path has issues (see above)"
}

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
case "${1:-up}" in
    up)
        log "range id=${RANGE_ID} | bridges: $(discover_bridges) | queue=${QUEUE_NUM}"
        sanitize_rules
        enable_bridge_netfilter
        install_steering
        start_snort
        verify
        ;;
    down)
        remove_steering
        systemctl stop "${SNORT_SERVICE}" && log "stopped ${SNORT_SERVICE}" || true
        ;;
    status)
        verify
        ;;
    *)
        die "usage: $0 [up|down|status]"
        ;;
esac
