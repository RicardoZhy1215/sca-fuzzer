.intel_syntax noprefix
# gs->hu control: identical to template-gs2hu.asm but with both indirect branches removed
# (attacker + victim fall through into their gadgets). purpose:
#   - if this still violates  -> the channel is the aliased shared-memory f+r, not branch injection
#   - if this is clean while baseline violates -> genuine vbti_gs->hu (an indirect branch is needed)

# ============================================================================
# vbti_gs->hu  (injection): guest-kernel attacker trains the btb;
#                           host-user victim mispredicts and leaks.
#
#   main     (host kernel)          : orchestrator only
#   attacker (guest kernel, observer): trains a btb entry, then measures via f+r.
#                                      data_alias=victim -> its guest memory gpa-collides
#                                      with the host-user victim's sandbox (config-gs2hu.yaml).
#   victim   (host user)            : runs its own indirect jmp; if the attacker-trained btb
#                                      entry survives into the host-user context, the victim
#                                      mispredicts to a disclosure gadget and leaves a footprint
#                                      the attacker then reads.
#
# mirror layout: attacker and victim use identical in-function offsets (same preamble length,
# same jmp offset, same gadget offset) to maximize btb source-pc collision on the low address
# bits. whether the entry actually crosses the host<->guest boundary is the empirical question.
#
# order (injection): attacker trains  ->  victim mispredicts/leaks  ->  attacker measures.
# no ibpb between training and the victim.
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
# warm-up: attacker starts measurement, returns.
.macro.set_h2g_target.attacker.function_attacker_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.0: nop qword ptr [rax + 0xff]


.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]

# ---- attacker trains the btb (guest kernel) ----
.macro.set_h2g_target.attacker.function_attacker_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.1: nop qword ptr [rax + 0xff]


.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]

# ---- no ibpb. run the host-user victim; it may mispredict to the trained target. ----
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

# ---- attacker measures ----
.macro.set_h2g_target.attacker.function_attacker_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- attacker (guest kernel, observer) ------------
.section .data.attacker

# round 0: warm-up — start measurement, return.
.function_attacker_0:
.macro.landing_h2g.attacker_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.attacker_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.attacker_0: nop qword ptr [rax + 0xff]
lfence


# round 1: train a btb entry with an indirect jmp at a fixed in-function offset.
.function_attacker_train:
.macro.landing_h2g.attacker_1: nop qword ptr [rax + 0xff]

# control: no indirect jmp; fall through into the gadget (btb never trained).
.macro.random_instructions.16.4.att_pre: nop qword ptr [rax + 0xff]

.function_attacker_gadget:
.macro.random_instructions.32.16.att_gadget: nop qword ptr [rax + 0xff]
lfence
.macro.switch_g2h.main.attacker_1: nop qword ptr [rax + 0xff]
lfence


# round 2: measurement end + f+r reload (reads the shared, aliased channel).
# zero the gprs before measuring so the final architectural register state is deterministic and
# matches the model (which carries registers across transitions as plain jumps). without this, the
# train-round register values survive in the model but are clobbered by the vm-exit/host/vm-enter
# in hardware -> every test case is wrongly filtered as an architectural mismatch. (f+r uses r14/r15,
# not these gprs, so zeroing them does not affect the measurement. mirrors the v2v observer, which
# re-initializes its registers each round.)
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
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove

# control: no indirect jmp in the victim either; fall through (no misprediction).
.macro.random_instructions.16.4.vic_pre: nop qword ptr [rax + 0xff]

.function_victim_safe:
# disclosure gadget: secret-dependent traffic in host-user context (r14 = victim data).
.macro.random_instructions.32.16.vic_gadget: nop qword ptr [rax + 0xff]
lfence  # noremove
.macro.switch_u2k.main.1: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
