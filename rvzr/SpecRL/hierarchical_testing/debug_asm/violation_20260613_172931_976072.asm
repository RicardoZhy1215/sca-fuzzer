.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 528 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2824 
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rsi] 
or rdx, rsi 
and rdi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax] 
and rdx, rsi 
and rax, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdi 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4880 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1784 
and rdi, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5024 
mov rbx, 4840 
inc rdx 
and rbx, rsi 
neg rdx 
xor rsi, rax 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1216 
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdi] 
xor rdi, rcx 
setz dl 
and rdi, rdx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2736 
and rbx, rdi 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6088 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 240 
mov rdx, 2920 
and rdi, rdi 
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi] 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rcx 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
xor rcx, rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6672 
and rdi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rdi] 
and rsi, rbx 
and rbx, rsi 
xor rcx, rdx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1904 
inc rdx 
cmp rbx, rdi 
xor rcx, rsi 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdx 
mov rax, 3088 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
xor rdi, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
