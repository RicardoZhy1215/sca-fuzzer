.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5112 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1200 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1528 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4040 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2680 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4248 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 432 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5160 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 904 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3672 
and rdi, rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 8152 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4688 
and rdi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2280 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1760 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2744 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7384 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 8152 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1560 
setl bl 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdi 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4256 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2152 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1096 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7248 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4584 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1216 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx 
and rdi, rsi 
and rdi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rdi] 
not rdx 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
xor rdi, rdi 
adc rcx, rax 
mov rdi, rcx 
sbb rdi, rdx 
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx] 
and rbx, rdi 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdi 
inc rax 
inc rdx 
inc rdi 
and rax, rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
