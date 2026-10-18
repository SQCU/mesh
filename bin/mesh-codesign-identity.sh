#!/bin/sh
# design/deliverables.md#p--packaging
set -eu
NAME="${MESH_CODESIGN_IDENTITY:-Mesh Bridge}"
KEYCHAIN=/Library/Keychains/System.keychain
if security find-identity -v -p codesigning "$KEYCHAIN" 2>/dev/null | grep -q "\"$NAME\""; then
  echo "codesign identity present: $NAME"; exit 0
fi
WORK=$(mktemp -d "${TMPDIR:-/var/tmp}/mesh-identity.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
cat >"$WORK/ext.cnf" <<EOF2
[ca]
default_ca=self
[self]
database=$WORK/index.txt
new_certs_dir=$WORK
serial=$WORK/serial
default_md=sha256
policy=named
x509_extensions=ext
unique_subject=no
[named]
commonName=supplied
[req]
distinguished_name=dn
prompt=no
[dn]
CN=$NAME
[ext]
keyUsage=critical,digitalSignature
extendedKeyUsage=critical,codeSigning
basicConstraints=critical,CA:false
EOF2
: >"$WORK/index.txt"; echo 01 >"$WORK/serial"
openssl req -new -newkey rsa:2048 -nodes -config "$WORK/ext.cnf" -keyout "$WORK/key.pem" -out "$WORK/request.pem" >/dev/null 2>&1
openssl ca -batch -selfsign -notext -config "$WORK/ext.cnf" -keyfile "$WORK/key.pem" -in "$WORK/request.pem" -enddate 99991231235959Z -out "$WORK/cert.pem" >/dev/null 2>&1
openssl pkcs12 -export -inkey "$WORK/key.pem" -in "$WORK/cert.pem" -name "$NAME" -passout pass:mesh -out "$WORK/identity.p12" >/dev/null 2>&1
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"
$SUDO security import "$WORK/identity.p12" -k "$KEYCHAIN" -P mesh -T /usr/bin/codesign -A >/dev/null
$SUDO security add-trusted-cert -d -r trustRoot -p codeSign -k "$KEYCHAIN" "$WORK/cert.pem"
security find-identity -v -p codesigning "$KEYCHAIN" | grep "\"$NAME\""
