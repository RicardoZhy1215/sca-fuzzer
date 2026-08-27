.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
imul rcx, rbx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx 
and rax, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rax 
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx] 
or rdx, rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdi 
and rdx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
