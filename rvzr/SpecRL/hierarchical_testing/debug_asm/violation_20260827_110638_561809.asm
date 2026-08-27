.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdi 
and rdi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rsi] 
setnz sil 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rcx 
inc rcx 
adc rcx, rdx 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1152 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax] 
imul rcx, rcx 
and rcx, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rcx] 
mov rdx, rsi 
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdx 
imul rcx, rax 
mov rbx, 816 
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
