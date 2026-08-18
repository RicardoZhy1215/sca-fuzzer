.intel_syntax noprefix

# ============================================================================
# V2V channel POSITIVE CONTROL — tests ONLY whether the EPT physical alias makes
# the guest observer's main page share the guest victim's frame. No BTB, no
# misprediction: the observer DIRECTLY reads [r14+0x100] (its main page) and
# encodes it. Under the alias that address resolves to the VICTIM's data.
#
#   - observer reads VICTIM's data  -> alias WORKS  (then a clean injection run
#                                       means the misprediction/window is the issue)
#   - observer reads its OWN data   -> alias BROKEN (set_extended_page_tables EPT
#                                       alias not taking effect)
#
# Decide with verify_v2v_channel.sh (model, which is alias-unaware, predicts the
# observer reads its OWN data; hardware disagrees iff the alias is live).
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.vmvictim.function_vmvictim_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vmvictim.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.vm.0:

.function_main_2:
    .macro.landing_g2h.main_2:
.macro.fault_handler:
.patch_placeholder_fault_handler:
    nop


# ----------------------------- Victim (guest user) --------------------------
# Just holds its input-initialized data in its main page (the shared frame).
.section .data.vmvictim
.function_vmvictim_0:
    .macro.landing_h2g.vmvictim_0:
    nop qword ptr [rax + 0xff]  # noremove
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_0:
    lfence


# ----------------------------- Observer (guest user) ------------------------
.section .data.vm
.function_vm_0:
    .macro.landing_h2g.vm_0:
    .macro.measurement_start.vm_0:
    # Read the observer's main page directly. Under the EPT alias this is the
    # VICTIM's frame -> rax ends up holding the victim's data (alias-unaware model
    # predicts the observer's own data => divergence iff the alias is live).
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x100]  # noremove   (victim data via EPT alias)
    and rax, 0b111111000000  # noremove
    mov rcx, qword ptr [r14 + rax]    # noremove   (encode into F+R)
    lfence  # noremove
    .macro.measurement_end.vm_0:
    .macro.switch_g2h.main.vm_0:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
