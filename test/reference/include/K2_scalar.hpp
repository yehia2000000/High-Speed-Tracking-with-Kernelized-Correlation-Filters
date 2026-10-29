#ifndef K2_SCALAR_H_ 
#define K2_SCALAR_H_

namespace ref 
{
    typedef enum { K2_ATAN2, K2_POLY } K2Method;    /* three ways to get the bin */

    extern const float K2_COS[9], K2_SIN[9];    /* cos, sin of k*20 deg, k = 0..8   (03) */
    extern const float K2_ATAN_C[6];            /* atan(a) ~ a * P(a*a) on [0, 1]  (02) */
    
    #define K2_HALF_PI    1.57079633f
    #define K2_PI         3.14159265f
    #define K2_TWO_PI     6.28318531f
    #define K2_9_OVER_PI  2.86478898f           /* radians -> units of 20 degrees   */

    /* scalar (k2_scalar.c) */
    void  k2_grad(const float *in, int P, int x, int y, float *gx, float *gy);
    float k2_mag(float gx, float gy);
    int   k2_bin_atan2(float gx, float gy);
    float k2_angle_poly(float gx, float gy);
    int   k2_bin_poly(float gx, float gy);
    void  k2_scalar(const float *in, int T, float *mag, uint8_t *bin, K2Method method);
    

};



#endif // !K2_SCALAR_H_ 



