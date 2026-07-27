/* Multi-model config acceptance: load_cfg() must parse the real dimension sets
 * of the roadmap models (MULTIMODEL_PLAN.md) without refusing or clamping:
 *
 *   - Kimi-K2   (1T): DeepseekV3ForCausalLM, n_group=1 — passes the GLM path
 *   - DeepSeek-V3 (671B): n_group=8, topk_group=4 — needs group routing
 *   - GLM-5.2 (744B): the shipped baseline, must stay exactly as before
 *
 * Dims below are copied from the models' real HF config.json files (fetched
 * 2026-07-26/27). The test writes each config to a temp dir, runs the REAL
 * load_cfg (include-glm.c pattern), and asserts every parsed field. This is
 * the guard that keeps dim generalization honest: a future hardcoded-GLM
 * assumption that rejects K2/DSv3 shapes fails here, not on a 600 GB model
 * download. */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail=0;
#define FAIL(fmt, ...) do { \
    fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); g_fail++; } while(0)
#define CHECK(c, field, got, want) do { \
    if((got)!=(want)) FAIL("%s: %s = %d, want %d", c, field, (int)(got), (int)(want)); } while(0)

static void write_cfg(const char *dir, const char *json){
    char p[512]; snprintf(p,sizeof(p),"%s/config.json",dir);
    FILE *f=fopen(p,"w"); if(!f){ perror(p); exit(1); }
    fputs(json,f); fclose(f);
}

int main(void){
    char dir[]="/tmp/coli_cfg_XXXXXX";
    if(!mkdtemp(dir)){ perror("mkdtemp"); return 1; }

    /* ---- Kimi K2 (moonshotai/Kimi-K2-Instruct, real dims) ---- */
    write_cfg(dir,
      "{\"hidden_size\":7168,\"num_hidden_layers\":61,\"num_attention_heads\":64,"
      "\"n_routed_experts\":384,\"num_experts_per_tok\":8,\"moe_intermediate_size\":2048,"
      "\"intermediate_size\":18432,\"first_k_dense_replace\":1,\"q_lora_rank\":1536,"
      "\"kv_lora_rank\":512,\"qk_nope_head_dim\":128,\"qk_rope_head_dim\":64,"
      "\"v_head_dim\":128,\"n_shared_experts\":1,\"vocab_size\":163840,"
      "\"n_group\":1,\"topk_group\":1,\"norm_topk_prob\":true,"
      "\"routed_scaling_factor\":2.827,\"scoring_func\":\"sigmoid\","
      "\"rms_norm_eps\":1e-6,\"eos_token_id\":1}");
    { Cfg c; load_cfg(&c,dir);
      CHECK("K2","hidden",c.hidden,7168);
      CHECK("K2","n_layers",c.n_layers,61);
      CHECK("K2","n_experts",c.n_experts,384);
      CHECK("K2","topk",c.topk,8);
      CHECK("K2","q_lora",c.q_lora,1536);
      CHECK("K2","kv_lora",c.kv_lora,512);
      CHECK("K2","qk_nope",c.qk_nope,128);
      CHECK("K2","qk_rope",c.qk_rope,64);
      CHECK("K2","v_head",c.v_head,128);
      CHECK("K2","n_group",c.n_group,1);
      CHECK("K2","score_func",c.score_func,0);
      CHECK("K2","norm_topk",c.norm_topk,1);
      if(c.routed_scale<2.82f||c.routed_scale>2.84f) FAIL("K2: routed_scale %g",c.routed_scale);
      /* absorb-path guard: kv_lora must fit the fixed qabs/clat buffers */
      if(c.kv_lora>512) FAIL("K2: kv_lora %d exceeds absorb buffers",c.kv_lora);
      if(c.qk_rope>256) FAIL("K2: qk_rope %d exceeds rope buffer",c.qk_rope); }

    /* ---- DeepSeek-V3 (deepseek-ai/DeepSeek-V3, real dims incl. groups) ---- */
    write_cfg(dir,
      "{\"hidden_size\":7168,\"num_hidden_layers\":61,\"num_attention_heads\":128,"
      "\"n_routed_experts\":256,\"num_experts_per_tok\":8,\"moe_intermediate_size\":2048,"
      "\"intermediate_size\":18432,\"first_k_dense_replace\":3,\"q_lora_rank\":1536,"
      "\"kv_lora_rank\":512,\"qk_nope_head_dim\":128,\"qk_rope_head_dim\":64,"
      "\"v_head_dim\":128,\"n_shared_experts\":1,\"vocab_size\":129280,"
      "\"n_group\":8,\"topk_group\":4,\"norm_topk_prob\":true,"
      "\"routed_scaling_factor\":2.5,\"scoring_func\":\"sigmoid\","
      "\"rms_norm_eps\":1e-6,\"eos_token_id\":1}");
    { Cfg c; load_cfg(&c,dir);
      CHECK("DSv3","n_experts",c.n_experts,256);
      CHECK("DSv3","n_group",c.n_group,8);
      CHECK("DSv3","topk_group",c.topk_group,4);
      CHECK("DSv3","n_heads",c.n_heads,128);
      CHECK("DSv3","first_dense",c.first_dense,3);
      /* group shape must be internally consistent: 8 groups of 32, top-8 fits 4x32 */
      if(c.n_experts%c.n_group) FAIL("DSv3: groups don't divide experts");
      if(c.topk > c.topk_group*(c.n_experts/c.n_group)) FAIL("DSv3: topk exceeds kept groups"); }

    /* ---- GLM-5.2 regression: the baseline shape stays exactly as parsed before ---- */
    write_cfg(dir,
      "{\"hidden_size\":6144,\"num_hidden_layers\":93,\"num_attention_heads\":64,"
      "\"n_routed_experts\":256,\"num_experts_per_tok\":8,\"moe_intermediate_size\":2048,"
      "\"intermediate_size\":12288,\"first_k_dense_replace\":3,\"q_lora_rank\":2048,"
      "\"kv_lora_rank\":512,\"qk_nope_head_dim\":192,\"qk_rope_head_dim\":64,"
      "\"v_head_dim\":256,\"n_shared_experts\":1,\"vocab_size\":154880,"
      "\"n_group\":1,\"topk_group\":1,\"norm_topk_prob\":true,"
      "\"routed_scaling_factor\":2.5,\"index_topk\":2048,\"index_n_heads\":32,"
      "\"index_head_dim\":128,\"rms_norm_eps\":1e-5,\"eos_token_id\":[151329,151336,151338]}");
    { Cfg c; load_cfg(&c,dir);
      CHECK("GLM","hidden",c.hidden,6144);
      CHECK("GLM","n_layers",c.n_layers,93);
      CHECK("GLM","q_lora",c.q_lora,2048);
      CHECK("GLM","qk_nope",c.qk_nope,192);
      CHECK("GLM","v_head",c.v_head,256);
      CHECK("GLM","n_group",c.n_group,1);
      CHECK("GLM","score_func",c.score_func,0);   /* absent field -> sigmoid */
      CHECK("GLM","index_topk",c.index_topk,2048);
      CHECK("GLM","n_stop",c.n_stop,3); }

    /* ---- V4-style scoring_func parses (dims stay engine-shaped: the CSA/HCA
     * attention is NOT supported yet — this only locks the router plumbing) ---- */
    write_cfg(dir,
      "{\"hidden_size\":7168,\"num_hidden_layers\":61,\"num_attention_heads\":128,"
      "\"n_routed_experts\":384,\"num_experts_per_tok\":6,\"moe_intermediate_size\":3072,"
      "\"intermediate_size\":18432,\"first_k_dense_replace\":1,\"q_lora_rank\":1536,"
      "\"kv_lora_rank\":512,\"qk_nope_head_dim\":128,\"qk_rope_head_dim\":64,"
      "\"v_head_dim\":128,\"n_shared_experts\":1,\"vocab_size\":129280,"
      "\"norm_topk_prob\":true,\"routed_scaling_factor\":2.5,"
      "\"scoring_func\":\"sqrtsoftplus\",\"rms_norm_eps\":1e-6,\"eos_token_id\":1}");
    { Cfg c; load_cfg(&c,dir);
      CHECK("V4ish","score_func",c.score_func,2);
      CHECK("V4ish","topk",c.topk,6);
      CHECK("V4ish","moe_inter",c.moe_inter,3072); }

    /* ---- Qwen3-235B-A22B (Qwen/Qwen3-235B-A22B, real dims): the GQA family.
     * Field map diverges: num_experts, head_dim, no q/kv_lora, no shared, no
     * dense layers; the engine reuses kv_lora/qk_rope as the K/V cache rows. */
    write_cfg(dir,
      "{\"model_type\":\"qwen3_moe\",\"hidden_size\":4096,\"num_hidden_layers\":94,"
      "\"num_attention_heads\":64,\"num_key_value_heads\":4,\"head_dim\":128,"
      "\"num_experts\":128,\"num_experts_per_tok\":8,\"moe_intermediate_size\":1536,"
      "\"intermediate_size\":12288,\"norm_topk_prob\":true,\"vocab_size\":151936,"
      "\"rms_norm_eps\":1e-6,\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":1000000.0},"
      "\"eos_token_id\":151645}");
    { Cfg c; load_cfg(&c,dir);
      CHECK("Qwen3","gqa",c.gqa,1);
      CHECK("Qwen3","oss",c.oss,0);
      CHECK("Qwen3","n_kv_heads",c.n_kv_heads,4);
      CHECK("Qwen3","head_dim",c.head_dim,128);
      CHECK("Qwen3","n_experts",c.n_experts,128);
      CHECK("Qwen3","first_dense",c.first_dense,0);
      CHECK("Qwen3","n_shared",c.n_shared,0);
      CHECK("Qwen3","score_func",c.score_func,1);
      CHECK("Qwen3","kv_lora (K row)",c.kv_lora,4*128);
      CHECK("Qwen3","qk_rope (V row)",c.qk_rope,4*128);
      CHECK("Qwen3","norm_topk",c.norm_topk,1);
      /* default rope: inv[0]=1, mscale=1 */
      if(c.gqa_inv[0]!=1.f || c.gqa_mscale!=1.f)
          FAIL("Qwen3 rope table: inv0=%g mscale=%g",c.gqa_inv[0],c.gqa_mscale); }

    /* ---- GPT-OSS-120B (openai/gpt-oss-120b, real dims): GQA + sinks/sliding/yarn.
     * Experts use intermediate_size; router is topk-then-softmax (score_func=3). */
    write_cfg(dir,
      "{\"model_type\":\"gpt_oss\",\"hidden_size\":2880,\"num_hidden_layers\":36,"
      "\"num_attention_heads\":64,\"num_key_value_heads\":8,\"head_dim\":64,"
      "\"num_local_experts\":128,\"num_experts_per_tok\":4,"
      "\"intermediate_size\":2880,\"sliding_window\":128,"
      "\"layer_types\":[\"sliding_attention\",\"full_attention\"],"
      "\"vocab_size\":201088,\"rms_norm_eps\":1e-5,"
      "\"rope_parameters\":{\"rope_type\":\"yarn\",\"rope_theta\":150000.0,\"factor\":32.0,"
      "\"beta_fast\":32.0,\"beta_slow\":1.0,\"truncate\":false,"
      "\"original_max_position_embeddings\":4096},"
      "\"eos_token_id\":200002}");
    { Cfg c; load_cfg(&c,dir);
      CHECK("OSS","gqa",c.gqa,1);
      CHECK("OSS","oss",c.oss,1);
      CHECK("OSS","n_experts",c.n_experts,128);
      CHECK("OSS","moe_inter=dense_inter",c.moe_inter,2880);
      CHECK("OSS","score_func",c.score_func,3);
      CHECK("OSS","swa[0] sliding",c.swa[0],128);
      CHECK("OSS","swa[1] full",c.swa[1],0);
      CHECK("OSS","swa[2] parity fallback (even=sliding)",c.swa[2],128);
      CHECK("OSS","swa[3] parity fallback (odd=full)",c.swa[3],0);
      /* yarn: mscale = 0.1*ln(32)+1, inv_freq[0]=1/theta^0=1 mixed by ramp */
      float want_ms=0.1f*logf(32.f)+1.f;
      if(fabsf(c.gqa_mscale-want_ms)>1e-5f)
          FAIL("OSS yarn mscale: %g want %g",c.gqa_mscale,want_ms);
      if(!(c.gqa_inv[0]<=1.f && c.gqa_inv[0]>0.f))
          FAIL("OSS yarn inv0 out of range: %g",c.gqa_inv[0]); }

    /* cleanup */
    { char p[512]; snprintf(p,sizeof(p),"%s/config.json",dir); remove(p); rmdir(dir); }

    /* ---- fmt=7 (FP4) container tagging: qt_resolve_fmt must discriminate the
     * O+1-scale FP4 layout from every neighboring format, and qt_bytes must
     * account the extra magic float ---- */
    { int O=24, I=512, gs=0;
      int64_t nib=(int64_t)O*((I+1)/2);
      /* FP4: int4-sized weights + (O+1) scales */
      int f=qt_resolve_fmt("fp4-test",O,I,nib,((int64_t)O+1)*4,&gs);
      if(f!=7||gs!=0) FAIL("fp4 resolve: fmt=%d gs=%d, want 7/0",f,gs);
      /* plain int4: same weights, O scales — must stay fmt=2 */
      f=qt_resolve_fmt("i4-test",O,I,nib,(int64_t)O*4,&gs);
      if(f!=2) FAIL("i4 resolve: fmt=%d, want 2",f);
      /* qt_bytes accounts O+1 scale floats for fmt=7 */
      { QT t; memset(&t,0,sizeof(t)); t.fmt=7; t.O=O; t.I=I;
        int64_t want=nib+((int64_t)O+1)*4;
        if(qt_bytes(&t)!=want) FAIL("fp4 qt_bytes: %lld want %lld",
                                    (long long)qt_bytes(&t),(long long)want); }
      /* dispatch reality: matmul_qt on an fmt=7 QT runs matmul_fp4 (same result
       * as calling the kernel directly) */
      { QT t; memset(&t,0,sizeof(t)); t.fmt=7; t.O=8; t.I=32;
        static uint8_t q[8*16]; static float s[9]; static float x[32], y1[8], y2[8];
        for(unsigned i=0;i<sizeof(q);i++) q[i]=(uint8_t)(i*37+11);
        for(int o=0;o<8;o++) s[o]=0.5f+0.1f*o;
        memcpy(&s[8],"FP4",4);
        for(int i=0;i<32;i++) x[i]=(float)(i%7)-3.f;
        t.q4=q; t.s=s;
        matmul_qt(y1,x,&t,1);
        matmul_fp4(y2,x,q,s,1,32,8);
        for(int o=0;o<8;o++) if(y1[o]!=y2[o]) FAIL("fp4 dispatch o=%d: %g vs %g",o,y1[o],y2[o]); }
      g_fail+=0; }

    printf("test_model_cfg: 6 model shapes, %d failure(s)\n", g_fail);
    if(g_fail) return 1;
    puts("test_model_cfg: ok");
    return 0;
}
