.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 78 # instrumentation
add rdi, rdx
mov rdi, rax
add rbx, rcx
mov rsi, rdi
add rcx, rbx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
jns .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rsi
add rbx, rax
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
mov rsi, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
