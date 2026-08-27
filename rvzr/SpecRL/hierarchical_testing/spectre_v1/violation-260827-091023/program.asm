.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
lea rdx, qword ptr [rbx + rdx + 1]
test rbx, rbx
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx
not rdx
and rdi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx
lea rax, qword ptr [rdi + rax + 1]
and rdi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2536
and rcx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
