.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi 
and rax, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rax] 
lea rbx, qword ptr [rdx + rbx + 1] 
and rdx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rbx 
and rdx, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rbx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax 
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdi] 
setz al 
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rcx 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdx 
adc rax, rdi 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx 
or rax, rbx 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rcx 
and rax, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rax] 
or rcx, rsi 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rax 
mov rbx, rcx 
and rsi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rsi] 
setb al 
and rdx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rdx] 
cmp rsi, rbx 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
