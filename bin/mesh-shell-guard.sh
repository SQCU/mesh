MESH_GUARD_ROOT="${${(%):-%x}:A:h:h}"
MESH_GUARD="$MESH_GUARD_ROOT/bin/mesh-kill-guard.sh"

mesh_guard_scope() { [[ "$PWD" == "$MESH_GUARD_ROOT" || "$PWD" == "$MESH_GUARD_ROOT"/* ]] }

mesh_guard_path() {
  local d="$MESH_GUARD_ROOT/bin/guard"
  path=( ${path:#$d} )
  mesh_guard_scope && path=( "$d" $path )
  return 0
}

mesh-guard-accept-line() {
  if mesh_guard_scope && "$MESH_GUARD" --match "$BUFFER"; then
    zle -I
    "$MESH_GUARD"
    BUFFER=""
    zle reset-prompt
  else
    zle .accept-line
  fi
}

kill() {
  mesh_guard_scope && "$MESH_GUARD" --match "kill $*" && { "$MESH_GUARD"; return 2; }
  builtin kill "$@"
}

autoload -Uz add-zsh-hook
add-zsh-hook chpwd mesh_guard_path
mesh_guard_path
zle -N accept-line mesh-guard-accept-line
