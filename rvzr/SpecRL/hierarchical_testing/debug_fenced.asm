.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -4 # instrumentation
lfence
add rdx, 6
lfence
cmp rbx, rdi
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
or byte ptr [r14 + rbx], 1 # instrumentation
lfence
mov ax, 1 # instrumentation
lfence
div byte ptr [r14 + rbx]
lfence
jnle .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
cmp rax, rbx
lfence
cmp rcx, rbx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
or byte ptr [r14 + rbx], 1 # instrumentation
lfence
mov ax, 1 # instrumentation
lfence
div byte ptr [r14 + rbx]
lfence
mov rsi, rdx
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], 7
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
