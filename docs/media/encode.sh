#!/usr/bin/env bash
# Turn the raw screen captures in docs/media/_originals/ into the images the README embeds.
#
# WHY THIS EXISTS
#   The raw captures total 138 MB; the largest is a single 46 MB GIF. GitHub will serve them,
#   but a README that pulls 138 MB is one most readers never see finish loading. This brings the
#   whole set to ~14 MB with the content still legible at README width.
#
# THE SETTING THAT MATTERS
#   `stats_mode=diff` + `diff_mode=rectangle`: weight the palette by what MOVES, and re-encode
#   only the changed rectangle of each frame. These pages are a mostly-static UI around one
#   animating canvas, so this is worth far more than tuning colour counts. Measured on
#   field-attention-arcs: 12.5 MB -> 1.8 MB at 900 px / 7 fps with no visible loss on the arcs.
#
# TWO EDITORIAL DECISIONS, NOT COMPRESSION ONES
#   * The six-iteration view ships as a STILL. Its content is six labelled tiles whose labels
#     carry the numbers ("iteration 4 - 656 lit - 37% shared with 1"); downscaling a 1460 px
#     six-up to GIF width makes exactly the informative part unreadable. A comparison wants a
#     still at full resolution.
#   * The 67 s flow-diagram capture is cut in two at 40 s, because it is two demonstrations:
#     the diagram animating token by token, and then clicking a neuron to trace its path. Split,
#     each can be captioned for what it actually shows.
#
# USAGE
#   ffmpeg on PATH (winget install Gyan.FFmpeg), then: bash docs/media/encode.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -d _originals ] || { echo "docs/media/_originals/ not present -- nothing to encode." >&2; exit 1; }

PAL="split[a][b];[a]palettegen=max_colors=%d:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"

clip () {  # src out fps width colours [start] [duration]
  local src=$1 out=$2 fps=$3 w=$4 c=$5 ss=${6:-} d=${7:-}
  ffmpeg -y -loglevel error ${ss:+-ss $ss} ${d:+-t $d} -i "_originals/$src.gif" \
    -vf "fps=$fps,scale=$w:-1:flags=lanczos,$(printf "$PAL" "$c")" -loop 0 "$out.gif"
  printf '  %-24s %7s\n' "$out.gif" "$(du -h "$out.gif" | cut -f1)"
}

still () {  # src out width [seek]
  ffmpeg -y -loglevel error ${4:+-ss $4} -i "_originals/$1" -vframes 1 \
    -vf "scale=$3:-1:flags=lanczos" -pix_fmt rgb24 "$2.png"
  printf '  %-24s %7s\n' "$2.png" "$(du -h "$2.png" | cut -f1)"
}

#     source                out                    fps  w    col  start  dur
clip  field-full-view       field-full-view          4  780   32
clip  field-attention-arcs  field-attention-arcs     7  900   48
clip  loop-walkthrough      loop-walkthrough         4  840   32
clip  loop-flow-diagram     loop-flow-diagram        3  820   24     0    40
clip  loop-flow-diagram     loop-neuron-trace        3  820   24    40    27

still field-six-iterations.gif  field-six-iterations  1460  20
still loop-overview.png         loop-overview         1100

echo
echo "total (excluding _originals): $(du -sh --exclude=_originals . | cut -f1)"
