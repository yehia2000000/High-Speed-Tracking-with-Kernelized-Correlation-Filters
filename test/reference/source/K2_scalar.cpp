
#include "K2_scalar.hpp"
#include "riscv_math.hpp"   // <cmath> is not available on bare metal
#include <cstdint>
 
/* Output pixel (x, y) sits at in[y+1][x+1]: the apron gives every pixel 4 neighbours. */
void ref::k2_grad(const float *in, int P, int x, int y, float *gx, float *gy) {
    const float *c = in + (size_t)(y + 1) * P + (x + 1);   
    *gx = c[1] - c[-1];                                 
    *gy = c[P] - c[-P];  
    
}
float ref::k2_mag(float gx, float gy) {
    return sqrtf(fmaf(gy, gy, gx * gx));  /* gx*gx rounded, then + gy*gy fused (= vfmacc) */
}

int ref::k2_bin_atan2(float gx, float gy) {
    float th = atan2f(gy, gx) * 57.29577951f;
    if (th < 0.0f) th += 360.0f;
    int b = (int)floorf(th / 20.0f + 0.5f);
    if (b >= 18) b -= 18;
    return b;
}

/* atan(a) ~ a * (c0 + c1 r + ... + c5 r^5), r = a*a, a in [0, 1]. Fitted: error ~2e-6 rad. */
const float ref::K2_ATAN_C[6] = {0.99997723f, -0.33262283f, 0.19354036f,
                            -0.11642645f, 0.05264731f, -0.01171912f};

/* Angle of (gx, gy) in [0, 2pi], built only from operations the vector unit has. */
float ref::k2_angle_poly(float gx, float gy) {
    float ax = fabsf(gx), ay = fabsf(gy);
    float mn = fminf(ax, ay), mx = fmaxf(ax, ay);
    float a  = (mx == 0.0f) ? 0.0f : mn / mx;      /* in [0, 1]; flat pixel -> 0 */
    float r  = a * a;
    float p  = K2_ATAN_C[5];
    for (int k = 4; k >= 0; k--)
        p = fmaf(p, r, K2_ATAN_C[k]);           /* Horner: p = p*r + c_k     */
    float t = a * p;                            /* atan(a), in [0, pi/4]     */
    if (ay > ax)  t = K2_HALF_PI - t;           /* other half of the octant  */
    if (gx < 0.0f) t = K2_PI - t;               /* left half-plane           */
    if (gy < 0.0f) t = K2_TWO_PI - t;           /* y < 0 half-plane          */
    return t;
}

int ref::k2_bin_poly(float gx, float gy) {
    float t = k2_angle_poly(gx, gy);
    int b = (int)fmaf(t, K2_9_OVER_PI, 0.5f);    /* t / 20deg + 0.5, then floor */
    return (b >= 18) ? 0 : b;
}

/* (03) Added at integration. The method KCFcpp's fhog uses: project the gradient on the
   9 directions k*20 deg and keep the largest |projection|; a negative one means the
   opposite direction, bin k + 9. Strict '>' keeps the lower bin on a tie, e.g. a vertical
   gradient (gx = 0) at exactly 90 deg goes to bin 4, where rounding 4.5 up gives 5.
   The table holds the exact floats fhog computes: cosf/sinf(k * (float)CV_PI / 9). */
const float ref::K2_COS[9] = { 1.0f,  0.939692616f,  0.766044438f,  0.49999997f,  0.173648104f,
                              -0.173648298f, -0.50000006f, -0.766044617f, -0.939692676f };
const float ref::K2_SIN[9] = { 0.0f,  0.342020154f,  0.642787635f,  0.866025448f, 0.984807789f,
                               0.98480773f,  0.866025388f,  0.642787457f,  0.342020005f };

int ref::k2_bin_dot(float gx, float gy) {
    float best = K2_COS[0] * gx + K2_SIN[0] * gy;
    int b = 0;
    for (int k = 0; k < 9; k++) {
        float d = K2_COS[k] * gx + K2_SIN[k] * gy;
        if (d > best)       { best = d;  b = k; }
        else if (-d > best) { best = -d; b = k + 9; }
    }
    return b;
}

void  ref::k2_scalar(const float *in, int T, float *mag, uint8_t *bin, K2Method method){
    int P = T + 2;
    for (int y = 0; y < T; y++) {
        for (int x = 0; x < T; x++) {
            float gx, gy;
            k2_grad(in, P, x, y, &gx, &gy);
            mag[y * T + x] = k2_mag(gx, gy);
            int b = (method == K2_ATAN2) ? k2_bin_atan2(gx, gy)
                  : (method == K2_POLY)  ? k2_bin_poly(gx, gy)
                                         : k2_bin_dot(gx, gy);
            bin[y * T + x] = (uint8_t)b;
        }
    }

}