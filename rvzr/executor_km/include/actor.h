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

#endif // _ACTOR_H_
