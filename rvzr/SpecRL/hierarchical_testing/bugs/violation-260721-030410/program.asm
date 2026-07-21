.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -112 # instrumentation
lea rsi, qword ptr [rdi + rsi + 1]
mov rbx, rdi
and rdi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rcx
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rbx
mov rdi, 5000
and rcx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdx
adc rsi, rdx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rbx
setb bl
imul rsi, rbx
setl al
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rax]
jle .bb_0.1
jmp .exit_0
.bb_0.1:
lea rdi, qword ptr [rsi + rdi + 1]
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rcx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rsi
and rcx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax]
or rcx, rbx
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rdi]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
setl dl
and rdi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rbx
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi]
sbb rsi, rcx
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx]
mov rdx, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
