.intel_syntax noprefix

# ============================================================================
# v2v cache-transfer probe. isolates one question: does a cache imprint made by
# the victim guest survive the vmexit/vmresume and reach the observer's reload?
#
# no btb, no misprediction. the victim architecturally imprints a secret-dependent
# line into the shared main frame; the observer primes (before) and reloads (after),
# with the victim's vmexit in between — the exact split the injection template has.
#
#   observer reload varies with secret -> cache survives the vmexit
#                                          => injection's problem is only the trigger
#   observer reload is constant         -> cache does not transfer across vmexit
#                                          => fundamental (e.g. l1d flush on vm exit)
# ============================================================================

.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
nop


# ----------------------------- victim (guest user) --------------------------
.section .data.vmvictim
.function_vmvictim_0:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove   (secret)
and rax, 0b111111000000  # noremove
mov rcx, qword ptr [r14 + rax]    # noremove   (architectural imprint into shared main page)
lfence  # noremove
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- observer (guest user) ------------------------
.section .data.vm
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence

.function_vm_measure:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
