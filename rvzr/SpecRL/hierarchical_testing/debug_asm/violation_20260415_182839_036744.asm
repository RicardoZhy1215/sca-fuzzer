.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 108 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
sbb rdi, 0 
add rdx, 0 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rax 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5 
cmp rsi, rcx 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdx 
mov rdx, 2 
sbb rbx, rdx 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
jp .bb_0.1 
jmp .exit_0 
.bb_0.1:
xor rdx, rbx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], 2 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 0 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rbx 
sbb rbx, 4 
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax 
sbb rsi, 2 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
