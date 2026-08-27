.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rcx 
setnz sil 
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rax] 
setl al 
setnz dil 
and rsi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi] 
test rbx, rcx 
setnz dl 
setnz cl 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdi 
setnz sil 
neg rsi 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
and rdx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rdx] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
or rax, rsi 
and rdx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdx] 
setl sil 
and rsi, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rsi] 
sub rdx, rbx 
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rax 
mov rdx, rsi 
mov rax, 2320 
setnz al 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
test rax, rsi 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx 
test rdx, rax 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdx 
not rax 
setb al 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
