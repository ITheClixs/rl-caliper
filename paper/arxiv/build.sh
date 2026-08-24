#!/usr/bin/env bash
# Assemble a self-contained arXiv submission tarball from the paper sources.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
out="$here/arxiv/submission"
rm -rf "$out" && mkdir -p "$out"

cp "$here/main.tex" "$here/macros.tex" "$here/refs.bib" "$out/"
mkdir -p "$out/sections" "$out/figures" "$out/tables"
cp "$here"/sections/*.tex "$out/sections/"
cp "$here"/figures/*.pdf "$out/figures/"
cp "$here"/tables/*.tex "$out/tables/"

# arXiv prefers a prebuilt bibliography
if [ -f "$here/build/main.bbl" ]; then cp "$here/build/main.bbl" "$out/"; fi

cd "$out"
tar czf ../submission.tar.gz .
echo "wrote $here/arxiv/submission.tar.gz"
tar tzf ../submission.tar.gz | head -20
