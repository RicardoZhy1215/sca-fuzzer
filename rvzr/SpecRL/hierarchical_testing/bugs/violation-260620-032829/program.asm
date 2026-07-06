.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rdx, 4080
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6248
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5928
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2224
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 7368
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1576
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6616
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6832
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4304
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 8048
and rdx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6928
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx]
xor rdi, rbx
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rbx
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rbx]
pextrq rax, xmm5, 0
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2016
not rax
jmp .bb_0.1
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4936
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3184
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 264
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7568
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 120
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 392
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 8176
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1760
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4560
pxor xmm2, xmm6
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2704
and rbx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
dec rax
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
