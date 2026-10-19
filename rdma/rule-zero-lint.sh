#!/bin/bash
# design/recovery.md#five-seconds: the rule-zero lint, static, for a reviewer.  rule-zero-lint.sh RDMA_DIR prints
# file:line for each finding and exits 1 on any of
#   - ibv_dealloc_pd, ibv_dereg_mr or ibv_close_device in the bridge's sources (the .c and .h files Makefile's mesh-flow
#     rule names): the kernel reclaims at exit (05e44eda);
#   - sched_yield inside a loop there: a waiting thread parks on a word;
#   - a time constant past MESH_PLATFORM_NS that decides, in any rdma C source (.c, .h, .m): a #define NAME_NS, _US, _MS
#     or _S, valued from its definition (one defined from others, their product and quotient), compared, made into a
#     deadline or a bound, or handed to a wait;
# and lists, failing nothing, a comparison that paces an emission (its stamp restamped to now on that line or the next)
# and every ibv_ call site of the bridge with its enclosing function, each to be read against the quiet window.
set -u
R=${1:?the rdma directory}
cd "$R" || exit 64
bridge=$(awk '/^mesh-flow:/{for(i=2;i<=NF;i++)if($i~/\.[ch]$/)printf "%s ",$i;exit}' Makefile 2>/dev/null)
[ -n "$bridge" ] || bridge="mesh-link.c mesh-program.c mesh-verbs.h mesh-bridge.h"
sources=$(ls *.c *.h *.m 2>/dev/null)
prog=$(cat <<'AWK'
function strip(s,   out, i, n, c, q) {
  out = ""; n = length(s); i = 1
  while (i <= n) {
    c = substr(s, i, 1)
    if (incomment) { if (c == "*" && substr(s, i + 1, 1) == "/") { incomment = 0; out = out " "; i += 2 } else i++; continue }
    if (c == "/" && substr(s, i + 1, 1) == "*") { incomment = 1; i += 2; continue }
    if (c == "/" && substr(s, i + 1, 1) == "/") break
    if (c == "\"" || c == "'") {
      q = c; i++
      while (i <= n && substr(s, i, 1) != q) { if (substr(s, i, 1) == "\\") i++; i++ }
      out = out q q; i++; continue
    }
    out = out c; i++
  }
  return out
}
function unit(name) { return name ~ /_NS$/ ? 1 : name ~ /_US$/ ? 1e3 : name ~ /_MS$/ ? 1e6 : 1e9 }
function evaluate(body,   n, t, i, v, op, x) {
  gsub(/U?INT(32|64)_C/, "", body); gsub(/[()]/, " ", body); gsub(/\*/, " * ", body); gsub(/\//, " / ", body)
  n = split(body, t, /[ \t]+/); v = 1; op = "*"; x = 0
  for (i = 1; i <= n; i++) {
    if (t[i] == "" || t[i] ~ /^(u?int[0-9]+_t|double|float|long|unsigned|int|time_t|size_t)$/) continue
    if (t[i] == "*" || t[i] == "/") { op = t[i]; continue }
    if (t[i] ~ /^[0-9][0-9.]*([eE][-+]?[0-9]+)?[uUlL]*$/) { sub(/[uUlL]+$/, "", t[i]); x = t[i] + 0 }
    else if (t[i] in raw) x = raw[t[i]]
    else return ""
    v = op == "*" ? v * x : v / x; x = 1
  }
  return x ? v : ""
}
function word(s, w) { return s ~ ("(^|[^A-Za-z0-9_])" w "([^A-Za-z0-9_]|$)") }
function seconds(v) { return sprintf("%g s", v / 1e9) }
function finding(kind, text) { printf "%s %s:%d: %s\n", kind, FILENAME, FNR, text; if (kind == "FAIL") failures++ }
function settle(restamped) {
  if (pending == "") return
  if (restamped) printf "pace %s (its stamp %s restamped to now)\n", pending, stamp
  else { printf "FAIL %s, a comparison that decides\n", pending; failures++ }
  pending = ""
}
FNR == 1 { incomment = 0; inmacro = 0; depth = 0; extern_c = 0; paren = 0; stmt = ""; fn = ""; if (pass == 2) settle(0) }
{
  raw_line = $0; code = strip($0)
  macro = inmacro || code ~ /^[ \t]*#/; inmacro = macro && raw_line ~ /\\[ \t]*$/
}
pass == 1 && code ~ /^[ \t]*#[ \t]*define[ \t]+[A-Za-z_][A-Za-z0-9_]*_(NS|US|MS|S)[ \t]/ {
  name = code; sub(/^[ \t]*#[ \t]*define[ \t]+/, "", name); body = name; sub(/[ \t].*$/, "", name); sub(/^[^ \t]+[ \t]+/, "", body)
  v = evaluate(body)
  if (v != "") { raw[name] = v; ns[name] = v * unit(name) }
  next
}
pass == 1 { next }
pass == 2 && FNR == 1 && !("MESH_PLATFORM_NS" in ns) { print "rule-zero-lint: no MESH_PLATFORM_NS among the definitions"; broken = 1; exit 2 }
pass == 2 && macro { next }
pass == 2 {
  if (pending != "" && code ~ /[^ \t]/) { nospace = code; gsub(/[ \t]/, "", nospace); x = index(nospace, stamp "=now"); settle(x && substr(nospace, x + length(stamp) + 4, 1) != "=") }
  plain = code; gsub(/->|<<|>>/, " ", plain)
  for (name in ns) {
    if (ns[name] <= ns["MESH_PLATFORM_NS"] || !word(code, name)) continue
    what = name " (" seconds(ns[name]) ") past MESH_PLATFORM_NS (" seconds(ns["MESH_PLATFORM_NS"]) ")"
    if (plain ~ /[<>]/) {
      stamp = ""
      if (match(code, /now(_[a-z]+\(\))?[ \t]*-[ \t]*[A-Za-z_][A-Za-z0-9_]*((->|\.)[A-Za-z_][A-Za-z0-9_]*)*/)) {
        stamp = substr(code, RSTART, RLENGTH); sub(/^now(_[a-z]+\(\))?[ \t]*-[ \t]*/, "", stamp)
      }
      pending = FILENAME ":" FNR ": " what; nospace = code; gsub(/[ \t]/, "", nospace)
      if (stamp == "") settle(0)
      else if (x = index(nospace, stamp "=now")) settle(substr(nospace, x + length(stamp) + 4, 1) != "=")
    } else if (code ~ /[A-Za-z_]*(wait|sleep)[A-Za-z_]*[ \t]*\(/) finding("FAIL", what " bounds a wait")
    else if (plain ~ /[^=!<>]=[^=]/) finding("FAIL", what " makes a deadline or a bound")
  }
  if (!(FILENAME in isbridge)) next
  if (code ~ /ibv_(dealloc_pd|dereg_mr|close_device)([^A-Za-z0-9_]|$)/) {
    match(code, /ibv_(dealloc_pd|dereg_mr|close_device)/); finding("FAIL", substr(code, RSTART, RLENGTH) " in the bridge")
  }
  n = length(code)
  for (i = 1; i <= n; i++) {
    c = substr(code, i, 1)
    if (c ~ /[A-Za-z_]/) {
      w = substr(code, i); match(w, /^[A-Za-z_][A-Za-z0-9_]*/); id = substr(w, 1, RLENGTH); rest = substr(w, RLENGTH + 1)
      if (id == "sched_yield") {
        inloop = word(stmt, "for") || word(stmt, "while") || word(stmt, "do")
        for (d = 1; d <= depth; d++) if (loop[d]) inloop = 1
        if (inloop) finding("FAIL", "sched_yield inside a loop (" (fn == "" ? "file scope" : fn) ")")
      }
      if (id ~ /^ibv_/ && rest ~ /^[ \t]*\(/) finding("call", (fn == "" ? "file scope" : fn) ": " id)
      stmt = stmt id; i += length(id) - 1; continue
    }
    if (c == "(") paren++
    else if (c == ")") paren--
    else if (c == "{") {
      if (depth == 0 && stmt ~ /extern[ \t]*""/) { extern_c++; stmt = ""; continue }
      depth++; loop[depth] = word(stmt, "for") || word(stmt, "while") || word(stmt, "do")
      if (depth == 1) {
        head = stmt; fn = ""; gsub(/__attribute__[ \t]*\(\([^()]*\)\)/, " ", head)
        if (index(head, "(") && !(index(head, "=") && index(head, "=") < index(head, "("))) {
          head = substr(head, 1, index(head, "(") - 1)
          if (match(head, /[A-Za-z_][A-Za-z0-9_]*[ \t]*$/)) { fn = substr(head, RSTART, RLENGTH); sub(/[ \t]+$/, "", fn) }
        }
      }
      stmt = ""; continue
    } else if (c == "}") { if (depth == 0 && extern_c) extern_c--; else if (depth > 0) depth--; if (depth == 0) fn = ""; stmt = ""; continue }
    else if (c == ";" && paren <= 0) { stmt = ""; paren = 0; continue }
    stmt = stmt c
  }
  stmt = stmt " "
}
END { if (broken) exit 2; if (pass == 2) settle(0); exit failures ? 1 : 0 }
AWK
)
echo "rule-zero-lint: $(pwd -P); the bridge: $bridge"
awk -v bridge_files="$bridge" 'BEGIN{n=split(bridge_files,b," ");for(i=1;i<=n;i++)isbridge[b[i]]=1}'"$prog" pass=1 $sources pass=1 $sources pass=2 $sources
status=$?
[ $status -eq 0 ] && echo "rule-zero-lint: clean" || echo "rule-zero-lint: findings above (exit $status)"
exit $status
