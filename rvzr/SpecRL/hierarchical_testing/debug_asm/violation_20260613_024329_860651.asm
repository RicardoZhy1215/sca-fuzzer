.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rax, rcx 
cmp rcx, rdi 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rsi 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
setl sil 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rcx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi 
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx 
and rdi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx 
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx] 
and rdi, rax 
and rcx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rcx] 
setl dl 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi 
dec rax 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rdx] 
and rsi, rbx 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rcx 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rcx 
test rdi, rax 
imul rdi, rdx 
and rcx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rcx] 
xor rdi, rax 
and rdi, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rdi] 
setb dl 
and rcx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rcx] 
setl al 
mov rax, 5288 
and rax, rcx 
adc rsi, rsi 
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx] 
dec rsi 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rcx 
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
