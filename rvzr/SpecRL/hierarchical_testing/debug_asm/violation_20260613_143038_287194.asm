.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rcx 
and rdx, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi 
adc rax, rax 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1880 
sbb rdx, rax 
adc rdx, rdx 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx 
and rax, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi] 
sub rdx, rax 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rsi 
and rax, rbx 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rsi 
inc rdx 
inc rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4936 
inc rsi 
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax] 
or rdx, rdi 
not rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
mov rdi, 1664 
inc rbx 
setz sil 
cmp rsi, rdi 
and rdi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdi] 
setz cl 
or rdx, rdi 
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2224 
sbb rdx, rdi 
and rax, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rdi] 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
