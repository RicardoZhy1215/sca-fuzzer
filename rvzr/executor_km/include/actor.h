/// File: Header describing actor metadata
///
// Copyright (C) Microsoft Corporation
// SPDX-License-Identifier: MIT

#ifndef _ACTOR_H_
#define _ACTOR_H_

#include <linux/types.h>

#define MAX_ACTORS 16

typedef uint64_t actor_id_t;
typedef uint64_t actor_mode_t;
typedef uint64_t actor_pl_t;

enum {
    MODE_HOST = 0,
    MODE_GUEST = 1,
};

enum {
    PL_KERNEL = 0,
    PL_USER = 1,
};

typedef struct {
    actor_id_t id;
    actor_mode_t mode;
    actor_pl_t pl;
    uint64_t data_permissions;
    uint64_t data_ept_properties;
    // [EXTENSION] id of the host actor whose sandbox this (guest) actor's memory physically
    // aliases to under hpa_gpa_collisions. 0 (=main) preserves the original behavior. This
    // reuses the trailing metadata word that was previously unused ("code_permissions").
    uint64_t alias_target;
} actor_metadata_t;

extern size_t n_actors;
extern actor_metadata_t *actors;

// [EXTENSION] Per-actor guest-virtual offset applied to the actor's DATA region only (bytes,
// page-aligned). 0 = stock behaviour: every guest maps its data at the same GVA, which is what
// makes two guests' data lines share a linear address. A non-zero delta breaks that sharing for
// one actor, which is how we test whether the linear-address match is a necessary condition for
// the cross-guest forwarding seen in violation-260817-090802.
// Set through /sys/rvzr_executor/guest_va_delta ("<actor_id> <delta_in_pages>").
extern uint64_t guest_va_delta[MAX_ACTORS];

#endif // _ACTOR_H_
