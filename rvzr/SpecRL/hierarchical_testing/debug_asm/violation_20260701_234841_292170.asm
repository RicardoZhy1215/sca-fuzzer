.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, rsi 
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
xor rdi, rax 
and rsi, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rsi] 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdi 
sub rdi, rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rsi 
setb dil 
and rdx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax 
adc rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 7888 
and rsi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx] 
sub rbx, rbx 
cmp rbx, rax 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rsi 
xor rbx, rbx 
adc rbx, rdx 
and rdx, rdx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 464 
and rcx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx] 
adc rdx, rdx 
jmp .bb_0.1 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi 
lea rax, qword ptr [rbx + rax + 1] 
setb dl 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
setb dl 
xor rcx, rsi 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rax 
xor rdx, rdx 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rcx] 
imul rsi, rsi 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdx 
imul rdi, rsi 
adc rax, rbx 
adc rdi, rdi 
or rdi, rax 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx 
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
