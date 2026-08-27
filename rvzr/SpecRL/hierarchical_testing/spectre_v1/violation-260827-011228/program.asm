.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
sbb rax, rdi
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx
and rbx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rax
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rcx
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdx
and rdx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rdx]
setb cl
and rdi, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdi
and rbx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rbx]
sub rbx, rsi
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi
lea rcx, qword ptr [rdx + rcx + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
