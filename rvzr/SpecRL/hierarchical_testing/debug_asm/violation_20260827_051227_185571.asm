.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx 
inc rax 
and rsi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx 
lea rdx, qword ptr [rsi + rdx + 1] 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdi 
and rdi, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rdi] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
lea rdi, qword ptr [rsi + rdi + 1] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi 
xor rdi, rax 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdi 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rax 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi 
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
