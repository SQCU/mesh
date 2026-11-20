#!/bin/sh
# builds libroute_ml.dylib: fabric.ml with the OCaml runtime linked in (no OCaml needed where it runs), its C face
set -e
cd "$(dirname "$0")"
ocamlopt -output-complete-obj -o fabric_ml.o fabric.ml
cc -O2 -I "$(ocamlopt -where)" -c fabric_stubs.c -o fabric_stubs.o
cc -dynamiclib fabric_ml.o fabric_stubs.o -lm -install_name @rpath/libroute_ml.dylib -o libroute_ml.dylib
rm -f fabric_ml.o fabric_stubs.o fabric.cm* fabric.o
