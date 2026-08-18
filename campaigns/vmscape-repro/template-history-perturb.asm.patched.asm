.intel_syntax noprefix

# ============================================================================
# vmscape-style cross-vm btb / bti residue test.
#
# layout:
#   main      (host kernel)    : orchestrates vm-enter / vm-exit
#   vmvictim  (guest kernel)   : "trains" a btb entry with an indirect call
#   vm        (guest kernel,   : observer; performs its own indirect call;
#              observer=true)    if btb is not flushed across vm-exit / -enter,
#                                its indirect call may transiently jump to the
#                                victim-trained target, touching cache lines
#                                that leak through the f+r channel.
# ============================================================================

# ----------------------------- hypervisor (host) ----------------------------
.section .data.main
.function_main_0:
# warm-up observer round (collect baseline measurement state)
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]


.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]

# ---- victim round: trains the btb ----
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]


.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]

# note: deliberately *no* ibpb / btb flush here.
# this is the heart of the vmscape assumption: the platform
# does not scrub indirect-branch predictor state across vm-exits.

.macro.set_h2g_target.vm.function_vm_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]

xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove

.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]


.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]

.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:

# tail observer round — closes out measurement cleanly
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- vm - victim ----------------------------------
.section .data.vmvictim
.function_vmvictim_0:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]

# random pre-amble — adds code distance so the indirect jmp's pc
# offset within this function matches the observer's (btb-collision).
# placed before the lea so it cannot clobber the jump target.
.macro.random_instructions.16.4.vv_pre: nop qword ptr [rax + 0xff]

# compute the victim's "secret" indirect-jump target.
# done immediately before `jmp` so random_instructions can't overwrite rax.
lea rax, qword ptr [rip + .function_vmvictim_1]  # noremove
jmp rax  # noremove                              # terminator: trains btb at this pc offset

# gadget lives in its own function so the indirect jmp above is the
# last instruction of its bb (required by revizor's parser).
.function_vmvictim_1:
# secret-dependent memory traffic. r14 is the data-sandbox base in revizor;
# random_instructions sprinkles loads/stores that depend on input bytes,
# leaving cache footprints the observer can detect.
.macro.random_instructions.32.16.vv_gadget: nop qword ptr [rax + 0xff]

lfence
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- vm - observer --------------------------------
.section .data.vm

# round 0: warm-up — start measurement, immediately return.
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence


# round 1: the actual observation. we perform an indirect call at the
# same code offset within this actor as the victim used (both functions
# start with landing_h2g + random_instructions of equal length, so the
# `call` lands at a matching low-pc). if btb residue exists, the call
# transiently retargets to vmvictim_gadget's address, touching its
# cache lines, which f+r then picks up.
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

.macro.random_instructions.16.4.vm_pre: nop qword ptr [rax + 0xff]

# ============================================================
# control variant: global branch history perturbation
# insert a deterministic, predictable chain of taken conditional
# branches just before the observer's indirect jmp. each branch
# shifts a known bit into the global branch-history register,
# overwriting whatever history pattern the victim might have
# imprinted. if the leak depends on a history-indexed predictor
# (shared between victim and observer through global history),
# the violation rate should drop sharply. if it doesn't, the
# shared predictor is more likely tag-based (e.g. ita-style).
# ============================================================
# r15 is held at 0 throughout; `test r15, r15` then sets zf=1, so
# every `jz` below is taken. each bb needs its own flag-defining
# instruction (revizor's flag-patch pass requires this), so we
# pair `test` + `jz` in every bb. 32 branches deep -> enough to
# fully overwrite the global branch-history register on zen
# (documented ~24-32 bit depth).
xor r15, r15      # noremove
test r15, r15     # noremove
jz .perturb_h01   # noremove
.perturb_h01:
test r15, r15     # noremove
jz .perturb_h02   # noremove
.perturb_h02:
test r15, r15     # noremove
jz .perturb_h03   # noremove
.perturb_h03:
test r15, r15     # noremove
jz .perturb_h04   # noremove
.perturb_h04:
test r15, r15     # noremove
jz .perturb_h05   # noremove
.perturb_h05:
test r15, r15     # noremove
jz .perturb_h06   # noremove
.perturb_h06:
test r15, r15     # noremove
jz .perturb_h07   # noremove
.perturb_h07:
test r15, r15     # noremove
jz .perturb_h08   # noremove
.perturb_h08:
test r15, r15     # noremove
jz .perturb_h09   # noremove
.perturb_h09:
test r15, r15     # noremove
jz .perturb_h10   # noremove
.perturb_h10:
test r15, r15     # noremove
jz .perturb_h11   # noremove
.perturb_h11:
test r15, r15     # noremove
jz .perturb_h12   # noremove
.perturb_h12:
test r15, r15     # noremove
jz .perturb_h13   # noremove
.perturb_h13:
test r15, r15     # noremove
jz .perturb_h14   # noremove
.perturb_h14:
test r15, r15     # noremove
jz .perturb_h15   # noremove
.perturb_h15:
test r15, r15     # noremove
jz .perturb_h16   # noremove
.perturb_h16:
test r15, r15     # noremove
jz .perturb_h17   # noremove
.perturb_h17:
test r15, r15     # noremove
jz .perturb_h18   # noremove
.perturb_h18:
test r15, r15     # noremove
jz .perturb_h19   # noremove
.perturb_h19:
test r15, r15     # noremove
jz .perturb_h20   # noremove
.perturb_h20:
test r15, r15     # noremove
jz .perturb_h21   # noremove
.perturb_h21:
test r15, r15     # noremove
jz .perturb_h22   # noremove
.perturb_h22:
test r15, r15     # noremove
jz .perturb_h23   # noremove
.perturb_h23:
test r15, r15     # noremove
jz .perturb_h24   # noremove
.perturb_h24:
test r15, r15     # noremove
jz .perturb_h25   # noremove
.perturb_h25:
test r15, r15     # noremove
jz .perturb_h26   # noremove
.perturb_h26:
test r15, r15     # noremove
jz .perturb_h27   # noremove
.perturb_h27:
test r15, r15     # noremove
jz .perturb_h28   # noremove
.perturb_h28:
test r15, r15     # noremove
jz .perturb_h29   # noremove
.perturb_h29:
test r15, r15     # noremove
jz .perturb_h30   # noremove
.perturb_h30:
test r15, r15     # noremove
jz .perturb_h31   # noremove
.perturb_h31:
test r15, r15     # noremove
jz .perturb_h32   # noremove
.perturb_h32:

# observer's "legitimate" indirect target. set just before jmp.
lea rbx, qword ptr [rip + .function_vm_3]  # noremove
jmp rbx  # noremove                        # architecturally hits .function_vm_3,
# transiently may hit function_vmvictim_1 via btb residue.

# legit target as its own function/bb (required by revizor parser).
.function_vm_3:
# generic post-jmp workload; includes loads so the f+r sweep has
# plenty of victim cache footprints to surface.
.macro.random_instructions.32.16.vm_post: nop qword ptr [rax + 0xff]

lfence  # noremove

.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence


# round 2: cleanup
.function_vm_2:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
