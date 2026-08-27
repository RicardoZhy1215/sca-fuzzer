.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
not rcx 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3640 
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
lea rcx, qword ptr [rbx + rcx + 1] 
setnz dil 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4472 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
and rcx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rbx] 
setl dl 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rbx 
and rdi, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rdi] 
test rbx, rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rsi] 
test rdi, rsi 
cmp rdi, rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6712 
imul rbx, rdx 
imul rdi, rcx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
imul rax, rax 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdi 
test rax, rdx 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rcx 
setb dil 
setl bl 
setz cl 
setz bl 
and rdx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax 
sub rdi, rsi 
and rdx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdx] 
xor rcx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
