.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx] 
and rdi, rax 
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi 
or rax, rdi 
and rbx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rbx] 
and rdi, rbx 
and rcx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
setl dl 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
sbb rbx, rcx 
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi 
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdi 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
