#!/bin/sh
# Runs kcf_video_qemu (the KCF flow with the RVV kernels) on a PGM sequence under qemu-riscv64,
# then encodes the frames it drew into an mp4 on the host.
#
#   scripts/kcf_video.sh BUILD_DIR SEQ_DIR [OUT_DIR] [kcf_video options]
#
#   BUILD_DIR  the cmake build folder (has example/kcf_video_qemu)
#   SEQ_DIR    PGM folder from scripts/kcf_sequence.py --pgm (images.txt, region.txt, groundtruth.txt)
#   OUT_DIR    where boxes, csv, raw video and mp4 go (default SEQ_DIR)
#   options    passed to the app, e.g. --compare, --scales 1, --frames 100
#
# Environment: VLEN (default 128), QEMU (default qemu-riscv64), FPS (default 30),
#              ZOOM (default 2: the video is drawn at frame size, ffmpeg enlarges it),
#              GIF (unset: none; set: also the README GIF there, scripts/readme_gif.sh).
#
# Example:
#   scripts/kcf_video.sh build images/kcf_seq_pgm /tmp/kcf_out --compare
#   -> /tmp/kcf_out/video_rvv_s3.mp4   red = ground truth, green = 3 scales, yellow = 1 scale
set -e

if [ $# -lt 2 ]; then
    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

BUILD=$1
SEQ=$2
shift 2
OUT=$SEQ
case "$1" in
    ""|-*) ;;
    *) OUT=$1; shift ;;
esac

VLEN=${VLEN:-128}
QEMU=${QEMU:-qemu-riscv64}
FPS=${FPS:-30}
ZOOM=${ZOOM:-2}
APP=$BUILD/example/kcf_video_qemu

[ -x "$APP" ] || { echo "no $APP: cmake --build $BUILD --target kcf_video_qemu"; exit 1; }
command -v ffmpeg > /dev/null || { echo "ffmpeg not found: sudo apt install ffmpeg"; exit 1; }
mkdir -p "$OUT"

LOG=$OUT/kcf_video.log
"$QEMU" -cpu "max,vlen=$VLEN" "$APP" "$SEQ" --out "$OUT" "$@" | tee "$LOG"

# the app prints: wrote <path>.rgb: <n> frames, rgb24, <W>x<H>
LINE=$(grep '^wrote .*\.rgb: ' "$LOG" | tail -1)
[ -n "$LINE" ] || { echo "no video written (--no-video?)"; exit 0; }
RAW=$(echo "$LINE" | sed 's/^wrote \(.*\.rgb\):.*/\1/')
SIZE=$(echo "$LINE" | sed 's/.* \([0-9]*x[0-9]*\)$/\1/')
MP4=${RAW%.rgb}.mp4

ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt rgb24 -s "$SIZE" -r "$FPS" -i "$RAW" \
       -vf "scale=trunc(iw*$ZOOM/2)*2:trunc(ih*$ZOOM/2)*2:flags=neighbor" \
       -c:v libx264 -pix_fmt yuv420p -crf 18 -movflags +faststart "$MP4"
# the GIF from the lossless frames: exact box colours, unlike a GIF made from the mp4
[ -z "$GIF" ] || sh "$(dirname "$0")/readme_gif.sh" "$RAW" "$GIF" "$SIZE"
rm -f "$RAW"
echo "video: $MP4"
