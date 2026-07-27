/* K2 (Kimi) tokenizer validation: converter + pre-tokenizer + BPE, no model
 * download. tests/tok_k2_tiny.json is built by the REAL converter
 * (tools/tiktoken_to_tokenizer_json.py) from a synthetic tiktoken.model
 * trained by tools/make_tok_k2_fixture.py, so merge recovery is exercised for
 * real; expected ids in tok_k2_cases.txt come from tiktoken.Encoding with the
 * verbatim Kimi pat_str on the same ranks — the exact engine K2 ships.
 * Guards the [\p{Han}]+ arm (incl. 々 and friends that are Han but Lm/Lo and
 * must NOT join letter runs), the &&[^\p{Han}] letter classes, the '/'-less
 * punctuation tail, contractions, digit groups, whitespace branches, and
 * added-token atomicity; round-trips every case. Family dispatch requires
 * \p{Han} in the tokenizer's own Split pattern, so cl100k (GLM) and o200k
 * stay on their own paths — covered by the GLM oracle and test_tok_o200k. */
#define _GNU_SOURCE
#include "../tok.h"

int main(void) {
    Tok T;
    tok_load(&T, "tests/tok_k2_tiny.json");
    if (!T.k2 || T.o200k) { fprintf(stderr, "test_tok_k2: family detect wrong (k2=%d o200k=%d)\n", T.k2, T.o200k); return 1; }
    FILE *f = fopen("tests/tok_k2_cases.txt", "rb");
    if (!f) { perror("tests/tok_k2_cases.txt"); return 1; }
    /* fgets, not getline: MinGW's UCRT lacks getline (windows job) */
    char line[8192];
    int pass = 0, tot = 0, dpass = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t nr = strlen(line);
        while (nr > 0 && (line[nr-1] == '\n' || line[nr-1] == '\r')) line[--nr] = 0;
        if (nr == 0) continue;
        char *tab = strchr(line, '\t'); if (!tab) continue;
        *tab = 0;
        const char *text = line, *idstr = tab + 1;
        char tbuf[4096]; int tn = 0;
        for (const char *q = text; *q && tn < 4095; q++) {
            if      (q[0]=='\\' && q[1]=='n')  { tbuf[tn++]='\n'; q++; }
            else if (q[0]=='\\' && q[1]=='t')  { tbuf[tn++]='\t'; q++; }
            else if (q[0]=='\\' && q[1]=='r')  { tbuf[tn++]='\r'; q++; }
            else if (q[0]=='\\' && q[1]=='\\') { tbuf[tn++]='\\'; q++; }
            else tbuf[tn++] = *q;
        }
        tbuf[tn] = 0;
        int exp[512], ne = 0;
        for (const char *q = idstr; *q; ) {
            while (*q == ',' || *q == ' ') q++;
            if (!*q) break;
            exp[ne++] = atoi(q);
            while (*q && *q != ',') q++;
        }
        int got[512]; int ng = tok_encode(&T, tbuf, tn, got, 512);
        int ok = (ng == ne);
        for (int i = 0; i < ng && ok; i++) ok = (got[i] == exp[i]);
        tot++; if (ok) pass++;
        char dec[8192]; int dn = tok_decode(&T, got, ng, dec, 8191);
        int drt = (dn == tn) && !memcmp(dec, tbuf, tn);
        if (drt) dpass++;
        if (!ok || !drt) {
            fprintf(stderr, "MISMATCH text=%s\n  exp(%d):", text, ne);
            for (int i = 0; i < ne; i++) fprintf(stderr, " %d", exp[i]);
            fprintf(stderr, "\n  got(%d):", ng);
            for (int i = 0; i < ng; i++) fprintf(stderr, " %d", got[i]);
            fprintf(stderr, "\n  decode_ok=%d\n", drt);
        }
    }
    fclose(f);
    fprintf(stderr, "test_tok_k2: %d/%d encode, %d/%d round-trip\n", pass, tot, dpass, tot);
    if (tot < 40 || pass != tot || dpass != tot) return 1;
    printf("test_tok_k2: ok\n");
    return 0;
}
