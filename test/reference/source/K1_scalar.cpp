#include "K1_scalar.hpp"
#include <cassert>
#include "riscv_math.hpp"   // <cmath> is not available on bare metal

void ref::k1_map(Roi r, int T, K1Map &m)
{
    float ww = 2.5f * r.tw;
    float wh = 2.5f * r.th;
    m.sx = ww / (float)T;
    m.sy = wh / (float)T;
    float x0 = r.cx - 0.5f * ww;
    float y0 = r.cy - 0.5f * wh;
    /* x(u) = x0 + (u + 0.5)*sx - 0.5 with u = j - 1 (apron)
       -> x(j) = j*sx + x_base                                                */
    m.x_base = x0 - 0.5f * m.sx - 0.5f;
    m.y_base = y0 - 0.5f * m.sy - 0.5f;
    m.P = T + 2;
}

void ref::k1_coord(float base, float s, int j, int n, int &i0, int &i1, float &fr)
{
    float x = std::fma((float)j, s, base);     /* j*s + base, one rounding (same as vfmadd) */
    if (x < 0.0f)  x = 0.0f;
    if (x > (float)(n - 1)) x = (float)(n - 1);
    int xi = (int)x;
    i0 = xi;
    i1 = (xi + 1 < n) ? xi + 1 : n - 1;
    fr = x - (float)xi;
}

static inline float k1_lerp(float a, float b, float f) { return std::fma(f, b - a, a); }

float ref::k1_bilinear(const Image<uint8_t, ImageType::GRAY>& in,
                       int x0, int x1, int y0, int y1, float fx, float fy)
{
    const uint8_t* r0 = in.GetPtr(0, y0);   // top source row
    const uint8_t* r1 = in.GetPtr(0, y1);   // bottom source row
    float top = k1_lerp(r0[x0], r0[x1], fx);
    float bot = k1_lerp(r1[x0], r1[x1], fx);
    return k1_lerp(top, bot, fy);
}

void ref::k1_scalar(const Image<uint8_t, ImageType::GRAY>& in, Roi r, int T, float* out)
{
    K1Map m;
    k1_map(r, T, m);
    const int P = m.P;
    const int W = in.Width();
    const int H = in.Height();
    for (int jy = 0; jy < P; jy++) {
        int y0, y1; float fy;
        k1_coord(m.y_base, m.sy, jy, H, y0, y1, fy);
        for (int jx = 0; jx < P; jx++) {
            int x0, x1; float fx;
            k1_coord(m.x_base, m.sx, jx, W, x0, x1, fx);
            float v = k1_bilinear(in, x0, x1, y0, y1, fx, fy);
            out[jy * P + jx] = v * (1.0f / 255.0f);
        }
    }
}