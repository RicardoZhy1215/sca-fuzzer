.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rdx] 
imul rsi, rdx 
or rdi, rbx 
xor rcx, rsi 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
inc rax 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx 
cmp rbx, rbx 
setb dil 
sub rcx, rdx 
and rax, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rax] 
test rdi, rdi 
and rsi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx] 
lea rsi, qword ptr [rbx + rsi + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4744 
adc rax, rcx 
lea rsi, qword ptr [rdi + rsi + 1] 
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rcx] 
lea rcx, qword ptr [rcx + rcx + 1] 
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx] 
neg rax 
and rbx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rcx 
dec rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2928 
neg rdx 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdx 
setb cl 
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rbx] 
mov rax, 5824 
and rbx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rbx] 
setl cl 
setb dil 
and rcx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rbx] 
sbb rsi, rbx 
and rcx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
