.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
setnz cl 
setl dil 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
test rcx, rcx 
setl bl 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
and rdi, rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
or rsi, rdi 
and rbx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rbx] 
cmp rsi, rbx 
mov rcx, 5160 
setl bl 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3544 
test rbx, rcx 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rcx 
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rcx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
