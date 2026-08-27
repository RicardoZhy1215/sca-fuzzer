.intel_syntax noprefix

# ----------------------------- kernel-mode actor (victim) -------------------
.section .data.main
.function_main_0:
# observer start
.macro.set_k2u_target.user.function_user_0: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_k2u.user.0: nop qword ptr [rax + 0xff]


.function_main_1:
.macro.landing_u2k.main_1: nop qword ptr [rax + 0xff]

# secret injection
.macro.random_instructions.64.32.main_1: nop qword ptr [rax + 0xff]

.macro.set_k2u_target.user.function_user_1: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_k2u.user.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_u2k.main_2: nop qword ptr [rax + 0xff]

.macro.fault_handler: nop qword ptr [rax + 0xff]
.macro.set_k2u_target.user.function_user_2: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_3: nop qword ptr [rax + 0xff]
.macro.switch_k2u.user.2: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_u2k.main_3: nop qword ptr [rax + 0xff]
nop

# ----------------------------- user-mode actor ------------------------------
.section .data.user
.function_user_0:
.macro.landing_k2u.user_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
.macro.switch_u2k.main.user_0: nop qword ptr [rax + 0xff]
lfence


.function_user_1:
.macro.landing_k2u.user_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
lfence

# secret retrieval
.macro.random_instructions.64.32.user_1: nop qword ptr [rax + 0xff]

# make sure the model doesn't attempt to go further than this point
lfence  # noremove

.macro.measurement_end.user_1: nop qword ptr [rax + 0xff]
.macro.switch_u2k.main.1: nop qword ptr [rax + 0xff]
lfence


.function_user_2:
.macro.landing_k2u.user_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.user_2: nop qword ptr [rax + 0xff]
.macro.switch_u2k.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
