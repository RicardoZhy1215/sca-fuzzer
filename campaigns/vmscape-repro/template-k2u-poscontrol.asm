.intel_syntax noprefix

# ============================================================================
# K2U POSITIVE CONTROL for the host<->host alias channel.
#
# No BTB, no misprediction. The USER (observer) DIRECTLY reads the aliased victim
# data and encodes it into its own cache footprint. This tests only one thing:
# does user.data_alias=kvictim actually make the user see the kernel victim's data?
#
#   - VIOLATES  -> the host alias works. Then aligned-clean means the BTB injection
#                  isn't landing (an address-alignment issue), not a broken channel.
#   - CLEAN     -> the host alias is NOT working (set_host_data_alias is broken);
#                  the whole K2U-BTI experiment is invalid until it's fixed.
#
# kvictim does nothing but hold its (input-initialized) data; the analyser varies
# the victim's input within a class, so if the user sees it, the footprint varies.
# ============================================================================

.section .data.main
.function_main_0:
    .macro.set_k2u_target.user.function_user_0:
    .macro.set_u2k_target.main.function_main_1:
    .macro.switch_k2u.user.0:

.function_main_1:
    .macro.landing_u2k.main_1:
    .macro.switch.kvictim.function_victim_1:

.function_main_2:
.macro.fault_handler:
    .macro.set_k2u_target.user.function_user_1:
    .macro.set_u2k_target.main.function_main_3:
    .macro.switch_k2u.user.1:

.function_main_3:
    .macro.landing_u2k.main_3:
    nop


# ----------------------------- User (host user, observer) -------------------
.section .data.user
.function_user_0:
    .macro.landing_k2u.user_0:
    .macro.measurement_start.user_0:
    .macro.switch_u2k.main.user_0:
    lfence

.function_user_1:
    .macro.landing_k2u.user_1:
    # Read the aliased (kernel-victim) data directly, then use the value as a
    # cache-line index -> footprint depends on the victim's data.
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x100]  # noremove   (victim data byte, via alias)
    and rax, 0b111111000000  # noremove            (cache-line index within main page)
    mov rcx, qword ptr [r14 + rax]    # noremove   (imprint into cache)
    lfence  # noremove
    .macro.measurement_end.user_1:
    .macro.switch_u2k.main.1:
    lfence


# ----------------------------- Victim (host kernel) -------------------------
.section .data.kvictim
.function_victim_1:
    nop qword ptr [rax + 0xff]  # noremove   (hold; data is input-initialized)
    lfence  # noremove
    .macro.switch.main.function_main_2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
