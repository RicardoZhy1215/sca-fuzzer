.intel_syntax noprefix

# ============================================================================
# guest-kernel (observer/attacker) reads host-kernel (victim) — the vm-escape / "vmscape" direction.
#
# unlike the u2k / both-guest templates (where a guest vmvictim holds the secret), here the host
# (main) is the victim: it runs the fuzzed "secret injection", so the only cross-actor secret is the
# hypervisor's own data. the guest (vm, observer) runs the fuzzed "secret retrieval" (a fault-driven
# transient window) and measures (f+r). a ct-ni violation here = the guest transiently read the host's
# data that it has no architectural access to (a guest->hypervisor leak).
#
# flow:  vm.measurement_start (arm f+r)  ->  host injects secret (main_1)  ->  vm.retrieval+measure_end
#
# reach / why hpa_gpa is required for this direction:
#   host (main) code+data live at raw vmalloc vas; guests live at gva 0x0. so host<->guest do not
#   share a linear address -> the l1d microtag will not collide across this boundary (unlike the
#   guest<->guest case, where both guests sit at gva 0x0 and the microtag does collide). this
#   direction therefore relies on x86_enable_hpa_gpa_collisions:true (the guest's gpa aliases the
#   host's hpa) so the guest reaches the host's l1d line via shared physical (foreshadow-vm-style),
#   not via the microtag. the config sets it. with aliasing off this direction is expected to go
#   quiet — that itself is the discriminating experiment.
# ============================================================================

# ----------------------------- hypervisor (host) = victim -------------------
.section .data.main
.function_main_0:
# 1) hand control to the observer guest so it can arm the f+r probe (measurement_start)
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
# 2) host = victim: the fuzzed "secret injection" runs here, in the host, filling the host's l1d
#    with its secret data. (main's data is writable -> these stores do not fault.)
.macro.random_instructions.64.32.main_1: nop qword ptr [rax + 0xff]
# 3) hand control to the observer guest for the "secret retrieval"
.macro.set_h2g_target.vm.function_vm_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]

xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.patch_placeholder:

.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
nop

# ----------------------------- guest - observer (attacker) ------------------
.section .data.vm
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence

.function_vm_1:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
# secret retrieval — fuzzed body. a store to the observer's read-only faulty page (writable:false
# in the config) raises a #pf; in that transient window younger loads transiently read the host's
# aliased l1d line and encode it into the f+r probe.
.macro.random_instructions.64.32.vm_1: nop qword ptr [rax + 0xff]
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence

.function_vm_2:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence

# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
