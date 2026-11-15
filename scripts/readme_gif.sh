#!/bin/sh
# Makes the GIF shown at the top of README.md from the tracking video of scripts/kcf_video.sh.
#
#   scripts/readme_gif.sh [IN] [OUT_GIF] [WxH]
#
#   IN       video_<tag>.rgb (raw rgb24 written by kcf_video_qemu; needs WxH), or the encoded
#            .mp4. Default out_video/video_rvv_s3.mp4.
#   OUT_GIF  default docs/kcf_video.gif
#   WxH      frame size of a raw .rgb input (kcf_video_qemu prints it), e.g. 320x240
#
# Environment: GIF_FPS (default 15), GIF_WIDTH (default 320 = the frame size before
#              kcf_video.sh's 2x zoom), GIF_GRAYS (default 29).
#
# kcf_video.sh calls this with the raw frames when GIF is set:
#   GIF=docs/kcf_video.gif sh scripts/kcf_video.sh build images/kcf_seq_pgm out_video --compare
#
# GitHub does not play an .mp4 from the repo inside a README, but it shows a GIF. The palette is
# fixed: GIF_GRAYS grays (the frames are gray) + the three box colours of kcf_video.cpp
# (red = ground truth, green = multi-scale, yellow = single scale), so the boxes keep their
# colours. From an mp4, hue=s=3 brings back the colour of thin box lines that yuv420p blurred;
# the raw frames need nothing. Needs ffmpeg and Pillow (pip install Pillow).
set -e

IN=${1:-out_video/video_rvv_s3.mp4}
OUT=${2:-docs/kcf_video.gif}
SIZE=$3
GIF_FPS=${GIF_FPS:-15}
GIF_WIDTH=${GIF_WIDTH:-320}
GIF_GRAYS=${GIF_GRAYS:-29}

[ -f "$IN" ] || { echo "no $IN: make it with scripts/kcf_video.sh"; exit 1; }
command -v ffmpeg > /dev/null || { echo "ffmpeg not found: sudo apt install ffmpeg"; exit 1; }

case "$IN" in
    *.rgb)
        [ -n "$SIZE" ] || { echo "a raw .rgb input needs its size, e.g. 320x240"; exit 1; }
        INPUT="-f rawvideo -pix_fmt rgb24 -s $SIZE -r 30"   # 30 = kcf_video.sh's default FPS
        BOOST=""
        ;;
    *)
        INPUT=""
        BOOST=",hue=s=3"
        ;;
esac

PAL=$(mktemp --suffix=.png)
trap 'rm -f "$PAL"' EXIT

python3 - "$PAL" "$GIF_GRAYS" << 'EOF'
import sys
from PIL import Image

n = int(sys.argv[2])
cols = [(round(i * 255 / (n - 1)),) * 3 for i in range(n)]   # grays
cols += [(235, 50, 40), (40, 230, 60), (255, 210, 0)]        # RED, GREEN, YELLOW in kcf_video.cpp
cols += [cols[0]] * (256 - len(cols))                        # paletteuse wants 16x16 entries
im = Image.new("RGB", (16, 16))
im.putdata(cols)
im.save(sys.argv[1])
EOF

mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC2086  # INPUT is a list of ffmpeg options
ffmpeg -hide_banner -loglevel error -y $INPUT -i "$IN" -i "$PAL" \
       -lavfi "fps=$GIF_FPS,scale=$GIF_WIDTH:-1:flags=area$BOOST[v];[v][1:v]paletteuse=dither=none:diff_mode=rectangle" \
       -loop 0 "$OUT"
echo "gif: $OUT ($(du -h "$OUT" | cut -f1))"
