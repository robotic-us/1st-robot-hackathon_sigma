#!/usr/bin/env bash
# Make a tiny local CA plus a server certificate for the iPad motion page.
# The private CA stays in .local-certs; install only sigma-ca.crt on the iPad.
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: $0 LAN_IP [CERT_DIR] [HOSTNAME]" >&2
  exit 2
fi

lan_ip=$1
cert_dir=${2:-.local-certs}
host_name=${3:-$(hostname)}
openssl_bin=${OPENSSL_BIN:-openssl}

mkdir -p "$cert_dir"
chmod 700 "$cert_dir"
umask 077

ca_key="$cert_dir/sigma-ca.key"
ca_cert="$cert_dir/sigma-ca.crt"
server_key="$cert_dir/server.key"
server_csr="$cert_dir/server.csr"
server_cert="$cert_dir/server.crt"

if [ ! -f "$ca_key" ] || [ ! -f "$ca_cert" ]; then
  "$openssl_bin" req -x509 -newkey rsa:2048 -nodes -sha256 -days 3650 \
    -keyout "$ca_key" -out "$ca_cert" \
    -subj "/CN=SIGMA Local iPad CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"
fi

"$openssl_bin" req -new -newkey rsa:2048 -nodes -sha256 \
  -keyout "$server_key" -out "$server_csr" \
  -subj "/CN=$lan_ip" \
  -addext "subjectAltName=IP:$lan_ip,IP:127.0.0.1,DNS:$host_name,DNS:localhost" \
  -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth"

"$openssl_bin" x509 -req -sha256 -days 825 \
  -in "$server_csr" -out "$server_cert" \
  -CA "$ca_cert" -CAkey "$ca_key" -CAcreateserial \
  -copy_extensions copy

chmod 600 "$ca_key" "$server_key"
chmod 644 "$ca_cert" "$server_cert"

echo "HTTPS certificate: $server_cert"
echo "HTTPS private key:  $server_key"
echo "Install on iPad:    $ca_cert"
echo "Server URL:         https://$lan_ip:8080/baby"
