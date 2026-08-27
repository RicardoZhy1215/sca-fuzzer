// microtag_probe.c — detect the AMD L1D way-predictor / µTag (Take A Way Collide+Probe).
//
// base and cand share the same page offset (=> same L1D set) but are on different pages (=> different
// physical tag). With only 2 lines in an 8-way set there is NO capacity eviction, so if probing base
// is SLOW after touching cand, the way-predictor treated them as the same line => same µTag.
// Sweeping cand over many pages maps which pages collide -> the µTag hash (and proves the predictor
// exists on this Zen4).
//
// Build:  gcc -O2 -o microtag_probe microtag_probe.c
// Run  :  taskset -c 3 ./microtag_probe [num_pages] [reps]    (perf_env.sh first for stable rdtsc)
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/mman.h>

static inline void  maccess(volatile void *p){ (void)*(volatile unsigned char *)p; }
static inline void  clflush(volatile void *p){ __asm__ volatile("clflush (%0)"::"r"(p):"memory"); }
static inline void  mfence(void){ __asm__ volatile("mfence":::"memory"); }
static inline void  lfence(void){ __asm__ volatile("lfence":::"memory"); }
static inline uint64_t rdtscp(void){ unsigned lo,hi; __asm__ volatile("rdtscp":"=a"(lo),"=d"(hi)::"rcx"); return ((uint64_t)hi<<32)|lo; }

static uint64_t probe(volatile void *p){          // time one access to p
    lfence(); uint64_t t0=rdtscp(); lfence();
    maccess(p);
    lfence(); uint64_t t1=rdtscp(); lfence();
    return t1-t0;
}

int main(int argc, char **argv){
    int core = 3;
    size_t PAGES = (argc>1)?strtoul(argv[1],0,0):16384;   // sweep range (default 64 MB)
    int    REPS  = (argc>2)?atoi(argv[2]):4000;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core,&s); sched_setaffinity(0,sizeof(s),&s);

    size_t SZ = PAGES*4096;
    volatile char *buf = mmap(0, SZ, PROT_READ|PROT_WRITE,
                              MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    if (buf==MAP_FAILED){ perror("mmap"); return 1; }
    memset((void*)buf,1,SZ);
    volatile char *base = buf;                    // page 0, offset 0

    // reference latencies: L1 hit vs full flush (miss)
    uint64_t hit=~0ULL, miss=~0ULL;
    for(int i=0;i<REPS;i++){ maccess(base); uint64_t t=probe(base); if(t<hit) hit=t; }
    for(int i=0;i<REPS;i++){ clflush(base); mfence(); uint64_t t=probe(base); if(t<miss) miss=t; }
    uint64_t thr = hit + (miss-hit)*2/3;          // ~2/3 toward miss => conservative collision flag
    printf("# core %d   L1-hit=%lu cyc   flush-miss=%lu cyc   collision-threshold=%lu cyc\n",
           core, hit, miss, thr);
    if (miss <= hit+2){ printf("# timing not discriminative (hit~miss). Run perf_env.sh, pin an idle core.\n"); }

    uint64_t *tim = malloc((PAGES)*sizeof(uint64_t));
    printf("# sweeping %zu pages x %d reps (same set, different page; fenced setup)...\n", PAGES, REPS);
    uint64_t *rep = malloc(REPS*sizeof(uint64_t));
    for(size_t k=1;k<PAGES;k++){
        volatile char *cand = base + k*4096;      // same offset => same set; different page => diff tag
        for(int i=0;i<REPS;i++){
            maccess(base); lfence();              // base -> predicted way for µTag(base)
            maccess(cand); lfence();              // if same µTag+set: overwrites base's way prediction
            rep[i]=probe(base);                   // probe base: slow iff the prediction was clobbered
        }
        for(int i=0;i<REPS;i++) for(int j=i+1;j<REPS;j++) if(rep[j]<rep[i]){uint64_t t=rep[i];rep[i]=rep[j];rep[j]=t;}
        tim[k]=rep[REPS/2];                       // MEDIAN per page (robust to probabilistic collisions)
    }
    free(rep);
    // distribution + top-slowest pages (bimodality = a µTag signal even if the penalty is small)
    uint64_t *sorted = malloc(PAGES*sizeof(uint64_t)); size_t n=0;
    for(size_t k=1;k<PAGES;k++) sorted[n++]=tim[k];
    for(size_t i=0;i<n;i++) for(size_t j=i+1;j<n;j++) if(sorted[j]<sorted[i]){uint64_t t=sorted[i];sorted[i]=sorted[j];sorted[j]=t;}
    printf("# probe(base) latency across %zu pages:  min=%lu  p50=%lu  p90=%lu  p99=%lu  max=%lu\n",
           n, sorted[0], sorted[n/2], sorted[n*9/10], sorted[n*99/100], sorted[n-1]);
    printf("# (L1-hit=%lu, miss=%lu)  -> if p99/max are near 'miss' at a REGULAR page stride, that's µTag.\n", hit, miss);
    printf("# 15 slowest pages:\n");
    for(int r=0;r<15;r++){
        uint64_t want=sorted[n-1-r];
        for(size_t k=1;k<PAGES;k++) if(tim[k]==want){ printf("   page %6zu (off 0x%08zx): %3lu cyc\n", k, k*4096, tim[k]); tim[k]=0; break; }
    }
    free(sorted);
    printf("# ---------------------------------------------------------------\n");
    printf("# READ: uniform latency (max ~ p50 ~ L1-hit) => NO way-predictor collisions => NOT µTag.\n");
    printf("#       a cluster of slow pages at a regular stride => µTag present (Take A Way).\n");
    free(tim);
    return 0;
}
