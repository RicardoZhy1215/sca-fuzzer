.intel_syntax noprefix

# ============================================================================
# guest-KERNEL (observer/attacker) reads host-KERNEL (victim) — the VM-escape / "vmscape" direction.
#
# Unlike the u2k / both-guest templates (where a GUEST vmvictim holds the secret), here the HOST
# (main) is the VICTIM: it runs the fuzzed "secret injection", so the only cross-actor secret is the
# hypervisor's own data. The GUEST (vm, observer) runs the fuzzed "secret retrieval" (a fault-driven
# transient window) and measures (F+R). A ct-ni violation here = the guest transiently read the host's
# data that it has NO architectural access to (a guest->hypervisor leak).
#
# Flow:  vm.measurement_start (arm F+R)  ->  HOST injects secret (main_1)  ->  vm.retrieval+measure_end
#
# REACH / why hpa_gpa is required for THIS direction:
#   Host (main) code+data live at RAW vmalloc VAs; guests live at GVA 0x0. So host<->guest do NOT
#   share a linear address -> the L1D microtag will NOT collide across this boundary (unlike the
#   guest<->guest case, where both guests sit at GVA 0x0 and the microtag DOES collide). This
#   direction therefore relies on x86_enable_hpa_gpa_collisions:true (the guest's GPA aliases the
#   host's HPA) so the guest reaches the host's L1D line via SHARED PHYSICAL (Foreshadow-VM-style),
#   not via the microtag. The config sets it. With aliasing OFF this direction is expected to go
#   quiet — that itself is the discriminating experiment.
# ============================================================================

# ----------------------------- Hypervisor (Host) = VICTIM -------------------
.section .data.main
.function_main_0:
    # 1) hand control to the observer guest so it can arm the F+R probe (measurement_start)
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vm.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    # 2) HOST = VICTIM: the fuzzed "secret injection" runs HERE, in the host, filling the host's L1D
    #    with its secret data. (main's data is writable -> these stores do NOT fault.)
    .macro.random_instructions.64.32.main_1:
    # 3) hand control to the observer guest for the "secret retrieval"
    .macro.set_h2g_target.vm.function_vm_1:
    .macro.set_g2h_target.main.function_main_2:

    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove
.patch_placeholder:

    .macro.switch_h2g.vm.1:

.function_main_2:
    .macro.landing_g2h.main_2:
.macro.fault_handler:
.patch_placeholder_fault_handler:
    .macro.set_h2g_target.vm.function_vm_2:
    .macro.set_g2h_target.main.function_main_3:
    .macro.switch_h2g.vm.2:

.function_main_3:
    .macro.landing_g2h.main_3:
    nop

# ----------------------------- Guest - Observer (attacker) ------------------
.section .data.vm
.function_vm_0:
    .macro.landing_h2g.vm_0:
    .macro.measurement_start:
    .macro.switch_g2h.main.vm_0:
    lfence

.function_vm_1:
    .macro.landing_h2g.vm_1:
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    # secret retrieval — fuzzed body. A store to the observer's read-only faulty page (writable:false
    # in the config) raises a #PF; in that transient window younger loads transiently read the host's
    # aliased L1D line and encode it into the F+R probe.
    .macro.random_instructions.64.32.vm_1:
    lfence  # noremove
    .macro.measurement_end.vm_1:
    .macro.switch_g2h.main.1:
    lfence

.function_vm_2:
    .macro.landing_h2g.vm_2:
    .macro.measurement_end.vm_2:
    .macro.switch_g2h.main.2:
    lfence

# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
