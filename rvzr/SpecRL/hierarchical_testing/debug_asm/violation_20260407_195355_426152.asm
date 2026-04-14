.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 102 # instrumentation
cmp rdx, rbx 
add rcx, 4 
sbb rax, rsi 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rdx 
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 0 
and rcx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 0 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 0 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdi 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx] 
cmp rbx, rax 
js .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax 
xor rdx, rdx 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 6 
add rbx, rax 
and rdi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rdi] 
add rdx, 0 
and rcx, 0b1111111111111 # instrumentation
sbb rax, qword ptr [r14 + rcx] 
cmp rsi, rcx 
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
