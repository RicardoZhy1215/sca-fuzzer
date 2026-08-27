.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rbx 
setb sil 
setnz sil 
and rdx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdx] 
imul rcx, rsi 
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6088 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdi 
lea rbx, qword ptr [rdi + rbx + 1] 
mov rdx, 7960 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdi 
and rax, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
