/* FP4 (e2m1) dequant + matmul oracle. Three contracts:
 *
 * 1. TABLE: fp4_dec must decode all 16 nibble codes to exactly the OCP MX
 *    e2m1 value set {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} (sign bit 3,
 *    exponent bits 2-1, mantissa bit 0).
 * 2. ROUND-TRIP: pack_fp4 -> matmul_fp4 on weights that are exactly
 *    representable (table values x arbitrary positive row scale) must
 *    reproduce the f32 matmul to float tolerance (the pack is lossless
 *    there, so the only error is dot-product rounding).
 * 3. SIMD == SCALAR: the AVX2 table-lookup path and the scalar tail must
 *    agree to 1 ULP-class tolerance for every alignment (I not multiple of
 *    16 exercises the mixed path), and the batched S>1 call must equal S=1
 *    calls row-by-row bit-exactly (independent (o,s) dots).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../quant.h"

static int g_fail=0;
#define FAIL(fmt, ...) do { \
    fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); g_fail++; } while(0)

static unsigned rng=0xC0FFEEu;
static float frand(void){ rng=rng*1664525u+1013904223u;
    return ((rng>>8)&0xFFFFFF)/(float)0xFFFFFF*2.f-1.f; }

int main(void){
    int cases=0;
    /* 1) decode table: all 16 codes, exact bit-level e2m1 semantics */
    { const float expect[16]={0,0.5f,1,1.5f,2,3,4,6, 0,-0.5f,-1,-1.5f,-2,-3,-4,-6};
      for(int c=0;c<16;c++){
          float got=fp4_dec((uint8_t)c);
          if(got!=expect[c] && !(c==8 && got==0.f))   /* -0.0 == 0.0 */
              FAIL("table[%d]: %g != %g",c,got,expect[c]);
          /* e2m1 bit semantics cross-check: sign=bit3, exp=bits2-1, man=bit0 */
          { int s=(c>>3)&1, e=(c>>1)&3, mn=c&1;
            float mag = e==0 ? 0.5f*mn : ldexpf(1.f+0.5f*mn,e-1);
            float ref = s? -mag : mag;
            if(got!=ref && !(got==0.f&&ref==0.f))
                FAIL("e2m1 bits[%d]: table %g vs formula %g",c,got,ref); }
      }
      cases+=16; }
    /* also: every byte value decodes both nibbles into the table (no OOB) */
    { for(int b=0;b<256;b++){ float lo=fp4_dec((uint8_t)(b&0xF)), hi=fp4_dec((uint8_t)(b>>4));
        if(!isfinite(lo)||!isfinite(hi)) FAIL("byte %02x decodes non-finite",b); }
      cases+=256; }
    /* 2) lossless round-trip on representable weights */
    { int O=16, I=64;
      float *w=malloc((size_t)O*I*sizeof(float));
      for(int o=0;o<O;o++){ float rs=0.1f+o*0.37f;   /* row scale */
          for(int i=0;i<I;i++){ int c=(o*31+i*7)%16;
              w[(size_t)o*I+i]=fp4_e2m1_tab[c]*rs; } }
      uint8_t *q=malloc((size_t)O*((I+1)/2));
      float *sc=malloc((size_t)O*sizeof(float));
      pack_fp4(w,q,sc,O,I);
      float x[64]; for(int i=0;i<I;i++) x[i]=frand();
      float y_q[16], y_f[16];
      matmul_fp4(y_q,x,q,sc,1,I,O);
      for(int o=0;o<O;o++){ double a=0;
          for(int i=0;i<I;i++) a+=(double)w[(size_t)o*I+i]*x[i];
          y_f[o]=(float)a; }
      for(int o=0;o<O;o++){
          float tol=2e-5f*(1.f+fabsf(y_f[o]));
          if(fabsf(y_q[o]-y_f[o])>tol)
              FAIL("roundtrip o=%d: %g vs %g",o,y_q[o],y_f[o]); }
      free(w); free(q); free(sc); cases+=O; }
    /* 3a) SIMD path == pure-scalar reference at every alignment */
    for(int I=1;I<=67;I++){
        int O=8;
        uint8_t *q=malloc((size_t)O*((I+1)/2));
        float *sc=malloc((size_t)O*sizeof(float));
        for(int o=0;o<O;o++){ sc[o]=0.25f+0.5f*o;
            for(int i=0;i<(I+1)/2;i++) q[(size_t)o*((I+1)/2)+i]=(uint8_t)((rng=rng*1664525u+1013904223u)>>16); }
        float *x=malloc((size_t)I*sizeof(float));
        for(int i=0;i<I;i++) x[i]=frand();
        float y[8];
        matmul_fp4(y,x,q,sc,1,I,O);
        for(int o=0;o<O;o++){
            double a=0; const uint8_t *w=q+(size_t)o*((I+1)/2);
            for(int i=0;i<I;i++){ uint8_t nib=(uint8_t)((w[i>>1]>>((i&1)*4))&0xF);
                a+=(double)x[i]*fp4_dec(nib); }
            float ref=(float)(a*sc[o]);
            float tol=3e-5f*(1.f+fabsf(ref));
            if(fabsf(y[o]-ref)>tol)
                FAIL("I=%d o=%d: %g vs ref %g",I,o,y[o],ref);
        }
        free(q); free(sc); free(x); cases++;
    }
    /* 3b) batched S>1 == per-row S=1 bit-exactly */
    { int O=24, I=48, S=5;
      uint8_t *q=malloc((size_t)O*((I+1)/2));
      float *sc=malloc((size_t)O*sizeof(float));
      for(size_t i=0;i<(size_t)O*((I+1)/2);i++) q[i]=(uint8_t)((rng=rng*1664525u+1013904223u)>>16);
      for(int o=0;o<O;o++) sc[o]=0.5f+0.1f*o;
      float *x=malloc((size_t)S*I*sizeof(float));
      for(int i=0;i<S*I;i++) x[i]=frand();
      float *yb=malloc((size_t)S*O*sizeof(float));
      float *yr=malloc((size_t)S*O*sizeof(float));
      matmul_fp4(yb,x,q,sc,S,I,O);
      for(int s=0;s<S;s++) matmul_fp4(yr+(size_t)s*O, x+(size_t)s*I, q, sc, 1, I, O);
      for(int i=0;i<S*O;i++)
          if(memcmp(&yb[i],&yr[i],4)) FAIL("batch[%d]: %g vs %g",i,yb[i],yr[i]);
      free(q); free(sc); free(x); free(yb); free(yr); cases+=S*O; }
    printf("test_fp4: %d cases run, %d failure(s)\n", cases, g_fail);
    if(g_fail) return 1;
    puts("test_fp4: ok");
    return 0;
}
