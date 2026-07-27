/* DeepSeek-V3 group-limited routing: route_group_mask() must reproduce the
 * reference semantics of DeepseekV3TopkRouter.get_topk_indices exactly:
 *
 *   scores_for_choice = sigmoid(logits) + e_score_correction_bias
 *   group_scores      = scores_for_choice.view(n_group, gsz).topk(2).sum(-1)
 *   keep the topk_group best groups, mask every other group's experts,
 *   then per-expert top-k runs on the masked scores.
 *
 * Strategy: drive the REAL route_group_mask (include-glm.c pattern) on
 * randomized score vectors and compare the surviving-expert SET and the
 * subsequent top-k selection against an independent brute-force reference
 * implemented straight from the formula above. Also asserts:
 *   - n_group==1 and topk_group>=n_group are exact no-ops (GLM path safety)
 *   - group tie-breaking is first-index (torch.topk first-occurrence)
 *   - every selected expert lies in a kept group, never a masked one
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail=0;
#define FAIL(fmt, ...) do { \
    fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); g_fail++; } while(0)

static unsigned rng_state=0x12345u;
static float frand(void){ rng_state=rng_state*1664525u+1013904223u;
    return ((rng_state>>8)&0xFFFFFF)/(float)0xFFFFFF*4.f-2.f; }

/* independent reference: group scores + keep set, no shared code with the impl */
static void ref_group_keep(const float *choice, int E, int n_group, int topk_group,
                           unsigned char *keep){
    int gsz=E/n_group;
    float gs[256];
    for(int g=0; g<n_group; g++){
        /* top-2 by full sort of a copy (obviously-correct reference) */
        float *tmp=malloc((size_t)gsz*sizeof(float));
        memcpy(tmp, choice+(size_t)g*gsz, (size_t)gsz*sizeof(float));
        for(int a=0;a<gsz;a++) for(int b=a+1;b<gsz;b++)
            if(tmp[b]>tmp[a]){ float t=tmp[a]; tmp[a]=tmp[b]; tmp[b]=t; }
        gs[g]=tmp[0]+(gsz>1?tmp[1]:0.f);
        free(tmp);
    }
    memset(keep,0,(size_t)n_group);
    for(int r=0;r<topk_group;r++){ int best=-1; float bv=-1e30f;
        for(int g=0;g<n_group;g++) if(!keep[g] && gs[g]>bv){ bv=gs[g]; best=g; }
        if(best<0) break; keep[best]=1; }
}

/* per-expert top-k on scores (first-index tie-break), shared shape with FASE A */
static void topk_select(const float *sc, int E, int K, int *idx){
    unsigned char used[4096];
    if(E<1 || E>4096) return;                 /* also silences -Wstringop-overflow */
    memset(used,0,(size_t)E);
    for(int kk=0;kk<K;kk++){ int best=-1; float bv=-1e30f;
        for(int e=0;e<E;e++) if(!used[e] && sc[e]>bv){ bv=sc[e]; best=e; }
        idx[kk]=best; if(best>=0) used[best]=1; }
}

int main(void){
    int cases=0;
    /* sweep: E, n_group, topk_group shapes incl. DeepSeek-V3 (256/8/4), V4-ish
     * (384/1), GLM (256/1), tiny degenerate groups */
    int shapes[][3]={{256,8,4},{256,8,1},{256,8,7},{64,4,2},{64,8,3},
                     {384,8,4},{16,4,2},{8,4,2},{8,8,4},{256,1,1},{64,2,1}};
    for(unsigned si=0; si<sizeof(shapes)/sizeof(shapes[0]); si++){
        int E=shapes[si][0], NG=shapes[si][1], TG=shapes[si][2];
        int K = E/NG < 8 ? E/NG : 8;   /* keep K within one group's size */
        for(int trial=0; trial<200; trial++){
            float *choice=malloc((size_t)E*sizeof(float));
            float *masked=malloc((size_t)E*sizeof(float));
            for(int e=0;e<E;e++) choice[e]=frand();
            if(trial%17==0)                       /* tie plateaus stress ties */
                for(int e=0;e<E;e++) choice[e]=(float)((e/3)%5)*0.25f;
            memcpy(masked,choice,(size_t)E*sizeof(float));
            route_group_mask(masked,E,NG,TG);
            unsigned char keep[256];
            ref_group_keep(choice,E,NG,TG,keep);
            int gsz=E/NG;
            /* 1) mask correctness: expert survives iff its group is kept */
            for(int e=0;e<E;e++){
                int kept = keep[e/gsz];
                if(kept && masked[e]!=choice[e])
                    FAIL("[E=%d NG=%d TG=%d t=%d] kept expert %d score changed",E,NG,TG,trial,e);
                if(!kept && masked[e]!=-1e30f)
                    FAIL("[E=%d NG=%d TG=%d t=%d] masked expert %d not -inf",E,NG,TG,trial,e);
            }
            /* 2) end-to-end: top-k on masked == top-k restricted to kept groups */
            int idx_impl[64], idx_ref[64];
            topk_select(masked,E,K,idx_impl);
            float *restricted=malloc((size_t)E*sizeof(float));
            for(int e=0;e<E;e++) restricted[e]=keep[e/gsz]?choice[e]:-1e30f;
            topk_select(restricted,E,K,idx_ref);
            for(int kk=0;kk<K;kk++)
                if(idx_impl[kk]!=idx_ref[kk])
                    FAIL("[E=%d NG=%d TG=%d t=%d] topk[%d]: impl %d != ref %d",
                         E,NG,TG,trial,kk,idx_impl[kk],idx_ref[kk]);
            /* 3) every selected expert is in a kept group */
            for(int kk=0;kk<K;kk++)
                if(idx_impl[kk]>=0 && !keep[idx_impl[kk]/gsz])
                    FAIL("[E=%d NG=%d TG=%d t=%d] selected expert %d from masked group",
                         E,NG,TG,trial,idx_impl[kk]);
            free(choice); free(masked); free(restricted);
            cases++;
        }
    }
    /* 4) no-op guarantees: n_group==1 and topk_group>=n_group leave scores untouched */
    { int E=256; float a[256], b[256];
      for(int e=0;e<E;e++) a[e]=b[e]=frand();
      route_group_mask(b,E,1,1);
      if(memcmp(a,b,sizeof(a))) FAIL("n_group=1 not a no-op");
      route_group_mask(b,E,8,8);
      if(memcmp(a,b,sizeof(a))) FAIL("topk_group==n_group not a no-op");
      cases+=2; }
    /* 5) deterministic tie case: all-equal scores must keep the FIRST topk_group
     * groups (first-index tie-break, matching torch.topk first occurrence) */
    { int E=64, NG=8, TG=3; float c[64];
      for(int e=0;e<E;e++) c[e]=1.0f;
      route_group_mask(c,E,NG,TG);
      int gsz=E/NG;
      for(int g=0;g<NG;g++){
          int expect_kept = g<TG;
          for(int i=0;i<gsz;i++){
              int is_kept = c[g*gsz+i]==1.0f;
              if(is_kept!=expect_kept)
                  FAIL("tie-break: group %d expected %s", g, expect_kept?"kept":"masked");
          }
      }
      cases++; }
    /* 6) route_score arms vs independent reference math (multi-model routers).
     * sigmoid: 1/(1+e^-x); softmax: e^xi/sum (stable); sqrtsoftplus (DSv4):
     * sqrt(log(1+e^x)) — double-precision reference, 1e-6 rel tolerance. */
    { int E=64; float x0[64], sc[64];
      for(int e=0;e<E;e++) x0[e]=frand()*8.f;       /* wide range incl. |x|>6 */
      x0[0]=88.f; x0[1]=-88.f;                       /* stability extremes */
      /* sigmoid */
      memcpy(sc,x0,sizeof(sc)); route_score(sc,E,0);
      for(int e=0;e<E;e++){ double r=1.0/(1.0+exp(-(double)x0[e]));
          if(fabs(sc[e]-r) > 1e-6*(1.0+fabs(r))) FAIL("sigmoid[%d]: %g vs %g",e,sc[e],r); }
      /* softmax */
      memcpy(sc,x0,sizeof(sc)); route_score(sc,E,1);
      { double mx=-1e30,sm=0; for(int e=0;e<E;e++) if(x0[e]>mx) mx=x0[e];
        for(int e=0;e<E;e++) sm+=exp((double)x0[e]-mx);
        double tot=0;
        for(int e=0;e<E;e++){ double r=exp((double)x0[e]-mx)/sm; tot+=sc[e];
            if(fabs(sc[e]-r) > 1e-5*(1.0+fabs(r))) FAIL("softmax[%d]: %g vs %g",e,sc[e],r); }
        if(fabs(tot-1.0)>1e-4) FAIL("softmax sum %g != 1",tot); }
      /* sqrtsoftplus */
      memcpy(sc,x0,sizeof(sc)); route_score(sc,E,2);
      for(int e=0;e<E;e++){ double sp=log1p(exp(-fabs((double)x0[e])))+(x0[e]>0?(double)x0[e]:0.0);
          double r=sqrt(sp);
          if(fabs(sc[e]-r) > 1e-5*(1.0+fabs(r))) FAIL("sqrtsoftplus[%d] x=%g: %g vs %g",e,x0[e],sc[e],r); }
      /* stability: no NaN/Inf anywhere */
      for(int f=0;f<3;f++){ memcpy(sc,x0,sizeof(sc)); route_score(sc,E,f);
          for(int e=0;e<E;e++) if(!isfinite(sc[e])) FAIL("score_func=%d[%d] not finite",f,e); }
      cases+=3; }
    /* 7) DSv4 hash routing lookup: table-driven selection, OOB refusal, dedup */
    { int V=16, K=4, E=8;
      int32_t tab[16*4];
      for(int t=0;t<V;t++) for(int k=0;k<K;k++) tab[t*K+k]=(int32_t)((t*7+k*3)%E);
      int idx[8];
      /* exact table rows come back for every in-range token */
      for(int t=0;t<V;t++){
          int n=hash_route_select(tab,V,K,E,t,idx);
          /* reference: dedup of the raw row, order-preserving */
          int ref[8], rn=0;
          for(int k=0;k<K;k++){ int e=(int)tab[t*K+k]; int dup=0;
              for(int j=0;j<rn;j++) if(ref[j]==e) dup=1;
              if(!dup) ref[rn++]=e; }
          if(n!=rn) FAIL("hash tok %d: n=%d ref=%d",t,n,rn);
          for(int j=0;j<n&&j<rn;j++) if(idx[j]!=ref[j])
              FAIL("hash tok %d idx[%d]: %d != %d",t,j,idx[j],ref[j]);
      }
      /* out-of-range tokens and NULL table select nothing */
      if(hash_route_select(tab,V,K,E,-1,idx)!=0) FAIL("hash: token -1 selected");
      if(hash_route_select(tab,V,K,E,V,idx)!=0) FAIL("hash: token V selected");
      if(hash_route_select(NULL,V,K,E,0,idx)!=0) FAIL("hash: NULL table selected");
      /* hostile table entries (OOB expert ids) are skipped, valid ones kept */
      { int32_t bad[4]={-3, 2, 99, 5};
        int n=hash_route_select(bad,1,4,E,0,idx);
        if(n!=2 || idx[0]!=2 || idx[1]!=5) FAIL("hash: OOB filter got n=%d",n); }
      cases+=V+4; }
    printf("test_group_route: %d cases run, %d failure(s)\n", cases, g_fail);
    if(g_fail) return 1;
    puts("test_group_route: ok");
    return 0;
}
