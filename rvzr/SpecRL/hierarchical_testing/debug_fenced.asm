.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
sbb rdi, rsi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovnle rdi, qword ptr [r14 + rsi]
lfence
sub rbx, rsi
lfence
sub rax, rdi
lfence
setnz cl
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovns rax, qword ptr [r14 + rcx]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rsi], 4344
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rsi]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rdi], rbx
lfence
setl bl
lfence
setz dil
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
add rcx, qword ptr [r14 + rdi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovle rbx, qword ptr [r14 + rdi]
lfence
adc rdx, rdi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmp rdx, rdx # instrumentation
lfence
cmovz rdx, qword ptr [r14 + rdi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
movzx rax, byte ptr [r14 + rdi]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovnl rdx, qword ptr [r14 + rsi]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
movzx rdx, byte ptr [r14 + rax]
lfence
adc rcx, rax
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovl rsi, qword ptr [r14 + rsi]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rdi]
lfence
test rax, rcx
lfence
and rax, 0b1111111111111 # instrumentation
lfence
cmovs rax, qword ptr [r14 + rax]
lfence
adc rbx, rax
lfence
mov rdx, rcx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], rax
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rbx]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmovns rbx, qword ptr [r14 + rsi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovnle rbx, qword ptr [r14 + rdi]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmp rsi, qword ptr [r14 + rsi]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovs rdx, qword ptr [r14 + rcx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmovbe rax, qword ptr [r14 + rbx]
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rax], rax
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rbx]
lfence
mov rcx, rdi
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rax], rdi
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovs rdi, qword ptr [r14 + rcx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
or rbx, 1 # instrumentation
lfence
cmovnz rbx, qword ptr [r14 + rbx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmovs rsi, qword ptr [r14 + rbx]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovs rcx, qword ptr [r14 + rcx]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
cmovle rdi, qword ptr [r14 + rdi]
lfence
and rax, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rax], rcx
lfence
adc rsi, rax
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rbx]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
cmovns rbx, qword ptr [r14 + rcx]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
movzx rdx, byte ptr [r14 + rsi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
movzx rcx, byte ptr [r14 + rdi]
lfence
and rbx, rdi
lfence
test rax, rdx
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
