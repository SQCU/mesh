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
cat >"$WORK/ext.cnf" <<EOF
[req]
distinguished_name=dn
x509_extensions=ext
prompt=no
[dn]
CN=$NAME
[ext]
keyUsage=critical,digitalSignature
extendedKeyUsage=critical,codeSigning
basicConstraints=critical,CA:false
EOF
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -config "$WORK/ext.cnf" -keyout "$WORK/key.pem" -out "$WORK/cert.pem" >/dev/null 2>&1
openssl pkcs12 -export -inkey "$WORK/key.pem" -in "$WORK/cert.pem" -name "$NAME" -passout pass:mesh -out "$WORK/identity.p12" >/dev/null 2>&1
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"
$SUDO security import "$WORK/identity.p12" -k "$KEYCHAIN" -P mesh -T /usr/bin/codesign -A >/dev/null
$SUDO security add-trusted-cert -d -r trustRoot -p codeSign -k "$KEYCHAIN" "$WORK/cert.pem"
security find-identity -v -p codesigning "$KEYCHAIN" | grep "\"$NAME\""
