.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rax 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rcx 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx 
or rax, rsi 
xor rcx, rcx 
not rdx 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rcx] 
lea rsi, qword ptr [rdx + rsi + 1] 
pxor xmm5, xmm7 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi] 
lea rdx, qword ptr [rsi + rdx + 1] 
pxor xmm2, xmm7 
and rax, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
