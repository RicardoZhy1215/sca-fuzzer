.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rbx] 
and rdi, rcx 
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi] 
test rcx, rcx 
setb dl 
imul rdx, rax 
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi] 
setl bl 
and rdx, rdi 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rdi 
adc rcx, rbx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax 
setb al 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
and rdi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx 
test rax, rax 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
dec rdi 
test rcx, rcx 
setnz al 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
