.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, -17 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdx
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rdi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rsi]
imul rdi, rbx
jns .bb_0.1
jmp .exit_0
.bb_0.1:
imul rsi, rcx
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rax
and rdx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdx]
setz dil
mov rdi, 7360
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
