/// File: Dispatch header that includes the guest page table definitions for the architecture
///
// Copyright (C) Microsoft Corporation
// SPDX-License-Identifier: MIT

#ifndef _GUEST_PAGE_TABLES_H_
#define _GUEST_PAGE_TABLES_H_

#include "hardware_desc.h"
#include "page_tables_common.h"
#include "sandbox_manager.h"

// =================================================================================================
// Memory layout
// =================================================================================================

// start of guest's physical memory; this is an arbitrary large aligned number
#define GUEST_P_MEMORY_START 0
#define GUEST_V_MEMORY_START 0x0ULL
// [EXTENSION] Number of last-level (l1) page tables per guest actor. Stock Revizor used exactly one
// (a 2 MB guest VA window). The microtag experiments need to place two actors' data at VAs whose XOR
// distance reaches the hash's higher bit pairings: v15^v20 needs 0x108000 (fits in 2 MB), but
// v16^v21 needs 0x210000 and v17^v22 needs 0x420000, both of which require a larger window.
// 4 tables = 8 MB, which reaches v15^v20, v16^v21 and v17^v22.
#define N_GUEST_L1_TABLES    4
#define GUEST_MEMORY_SIZE    (N_GUEST_L1_TABLES * 512 * 4096)

// =================================================================================================
// Extended page tables
// =================================================================================================
#if defined(ARCH_X86_64)

// Kernel Constant Compatibility
#ifndef VMX_BASIC_MEM_TYPE_WB
#define VMX_BASIC_MEM_TYPE_WB 6LLU
#endif

// Memory layout within the guest memory
typedef struct {
    pte_t_ l1[N_GUEST_L1_TABLES][ENTRIES_PER_PAGE]; // PTs, one per 2 MB of guest VA
    pdte_t l2[ENTRIES_PER_PAGE];  // PDT
    pdpte_t l3[ENTRIES_PER_PAGE]; // PDPT
    pml4e_t l4[ENTRIES_PER_PAGE]; // PML4
} actor_page_table_t;

typedef struct {
    epte_t_ l1[ENTRIES_PER_PAGE];  // EPT PT
    epdte_t l2[ENTRIES_PER_PAGE];  // EPT PDT
    epdpte_t l3[ENTRIES_PER_PAGE]; // EPT PDPT
    epml4e_t l4[ENTRIES_PER_PAGE]; // EPT PML4
} actor_ept_t;

typedef struct {
    uint8_t entries[PAGE_SIZE];
} __attribute__((packed)) actor_gdt_t;

// Guest memory layout; it is identical for both physical and virtual memory
typedef struct {
    util_t util;
    actor_data_t data;
    actor_code_t code;
    uint8_t vmlaunch_page[PAGE_SIZE];
    actor_gdt_t gdt;
    actor_page_table_t guest_page_tables;
} __attribute__((packed)) guest_memory_t;

// Translation from virtual to guest and host physical addresses
typedef struct {
    uint64_t hpa;
    uint64_t gpa;
    void *gva;
    void *hva;
} __attribute__((packed)) hgpa_t;

// Specialized translation data structure to speed up virtual-to-physical translations
typedef struct {
    hgpa_t util[sizeof(util_t) / PAGE_SIZE];
    hgpa_t data[sizeof(actor_data_t) / PAGE_SIZE];
    hgpa_t code[sizeof(actor_code_t) / PAGE_SIZE];
    hgpa_t vmlaunch_page[1];
    hgpa_t gdt[1];
    hgpa_t guest_page_tables[N_GUEST_L1_TABLES + 3];
} __attribute__((packed)) guest_memory_translations_t;

extern eptp_t *ept_ptr;

#elif defined(ARCH_ARM)

// nothing here yet

#endif // ARCH_ARM

// =================================================================================================
// Public interfaces
// =================================================================================================
int dbg_dump_guest_page_tables(int actor_id);
int dbg_dump_ept(int actor_id);

int map_sandbox_to_guest_memory(void);

void set_faulty_page_guest_permissions(void);
void restore_faulty_page_guest_permissions(void);

void set_faulty_page_ept_permissions(void);
void restore_faulty_page_ept_permissions(void);

int allocate_guest_page_tables(void);
void free_guest_page_tables(void);

#endif // _GUEST_PAGE_TABLES_H_
