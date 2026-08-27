.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rax] 
setb sil 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi 
neg rdi 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx 
sub rcx, rcx 
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx] 
mov rdi, 4760 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx 
and rsi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx 
and rdi, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
