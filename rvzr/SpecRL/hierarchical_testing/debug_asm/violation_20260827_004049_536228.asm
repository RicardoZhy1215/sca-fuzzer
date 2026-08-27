.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rcx] 
or rcx, rax 
and rdx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rdx] 
xor rcx, rbx 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rcx 
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi 
and rcx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rcx] 
neg rcx 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rax 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
and rdx, rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rsi 
setnz bl 
and rsi, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rdx] 
setb dil 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rax 
and rsi, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rax] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
