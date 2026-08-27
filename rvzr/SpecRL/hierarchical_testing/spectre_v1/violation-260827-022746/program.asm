.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rcx]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
cmp rbx, rbx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7224
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rcx
and rdx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx
xor rcx, rsi
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 568
setl sil
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
lea rax, qword ptr [rdi + rax + 1]
sbb rax, rdx
or rdx, rsi
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
inc rsi
neg rcx
mov rdi, rbx
and rdi, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax
test rdi, rsi
inc rcx
and rbx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdi
setnz sil
setnz cl
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rsi]
mov rcx, 3272
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
