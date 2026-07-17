.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 70 # instrumentation
test rdi, rcx 
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax] 
setl bl 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rsi 
jz .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdx 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
not rax 
mov rsi, 5592 
and rcx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi 
lea rbx, qword ptr [rdi + rbx + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
