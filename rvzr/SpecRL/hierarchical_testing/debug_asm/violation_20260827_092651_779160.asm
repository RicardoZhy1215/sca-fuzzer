.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7064 
test rdi, rdi 
and rdi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdi] 
lea rdx, qword ptr [rsi + rdx + 1] 
lea rcx, qword ptr [rcx + rcx + 1] 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rbx 
lea rax, qword ptr [rsi + rax + 1] 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
and rdi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdi] 
setb bl 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi 
and rsi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rsi] 
test rdx, rdx 
mov rdx, 696 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdx] 
sub rdi, rbx 
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rdx] 
or rax, rsi 
adc rax, rcx 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
imul rax, rdi 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi 
xor rcx, rax 
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rcx 
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rbx 
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rcx] 
test rdx, rsi 
and rdx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
