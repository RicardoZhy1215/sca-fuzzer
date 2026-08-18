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

# control variant: pc-misalignment.
# observer's pre-amble is 32 random insns instead of 16, shifting
# the indirect `jmp` to a different pc offset within the actor.
# victim still uses 16 insns (see vv_pre above) -> the two indirect
# jmps now sit at different low-pc offsets, so they should not
# collide in any pc-indexed predictor table (btb / bhb).
# if the leak depends on pc-indexed cross-actor predictor sharing,
# this should kill it.
.macro.random_instructions.32.8.vm_pre: nop qword ptr [rax + 0xff]

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
