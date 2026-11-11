#!/usr/bin/env bash
# Builds a double-blind copy in ../anonymous_copy: no names, no git history.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../anonymous_copy
rm -rf "$OUT"; mkdir "$OUT"
tar --exclude=.git --exclude=archive --exclude=make_anonymous.sh --exclude=__pycache__ -cf - . | tar -xf - -C "$OUT"
sed -i '/^Maintainer:/d; /double-blind review copy/d' "$OUT/README.md"
sed -i 's/Copyright (c) 2026 Siba Mishra/Copyright (c) 2026 Anonymous Authors/' "$OUT/LICENSE"
grep -rniE "siba|mishra|patnaik|jjsmp2|NPatnaik|cgu|raman" "$OUT" --exclude-dir=.git -l || echo "no author strings found"
