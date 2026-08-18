.intel_syntax noprefix

# ============================================================================
# v2v channel positive control — tests only whether the ept physical alias makes
# the guest observer's main page share the guest victim's frame. no btb, no
# misprediction: the observer directly reads [r14+0x100] (its main page) and
# encodes it. under the alias that address resolves to the victim's data.
#
#   - observer reads victim's data  -> alias works  (then a clean injection run
#                                       means the misprediction/window is the issue)
#   - observer reads its own data   -> alias broken (set_extended_page_tables ept
#                                       alias not taking effect)
#
# decide with verify_v2v_channel.sh (model, which is alias-unaware, predicts the
# observer reads its own data; hardware disagrees iff the alias is live).
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
nop


# ----------------------------- victim (guest user) --------------------------
# just holds its input-initialized data in its main page (the shared frame).
.section .data.vmvictim
.function_vmvictim_0:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
nop qword ptr [rax + 0xff]  # noremove
lfence  # noremove
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- observer (guest user) ------------------------
.section .data.vm
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
# read the observer's main page directly. under the ept alias this is the
# victim's frame -> rax ends up holding the victim's data (alias-unaware model
# predicts the observer's own data => divergence iff the alias is live).
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x100]  # noremove   (victim data via ept alias)
and rax, 0b111111000000  # noremove
mov rcx, qword ptr [r14 + rax]    # noremove   (encode into f+r)
lfence  # noremove
.macro.measurement_end.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
