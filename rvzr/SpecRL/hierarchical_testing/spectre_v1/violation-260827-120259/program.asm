.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rax]
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
and rcx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx
and rdx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2808
and rsi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rsi]
mov rdx, 248
cmp rsi, rax
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
mov rsi, 856
lea rdx, qword ptr [rsi + rdx + 1]
setl sil
and rsi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rsi]
setnz cl
and rax, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rax]
and rdx, rax
and rax, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3872
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 808
inc rcx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6512
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6552
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rsi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
