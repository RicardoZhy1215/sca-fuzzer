.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx]
mov rcx, 3432
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
setb al
setl cl
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
not rcx
not rbx
neg rbx
xor rsi, rsi
and rax, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
