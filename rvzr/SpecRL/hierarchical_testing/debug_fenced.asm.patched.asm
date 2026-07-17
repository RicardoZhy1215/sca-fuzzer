.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, 70 # instrumentation
lfence
test rdi, rcx
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mov rdi, qword ptr [r14 + rax]
lfence
setl bl
lfence
or rax, 1 # instrumentation
lfence
and rdx, rax # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rax
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rbx]
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rsi]
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock cmpxchg qword ptr [r14 + rbx], rsi
lfence
jz .bb_0.1
jmp .exit_0
.bb_0.1:
lfence
and rsi, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rsi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmovle rcx, qword ptr [r14 + rbx]
lfence
and rax, 0b1111111111111 # instrumentation
lfence
add rdx, qword ptr [r14 + rax]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
cmovs rsi, qword ptr [r14 + rbx]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
movsx rdx, byte ptr [r14 + rdi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], rdi
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock and qword ptr [r14 + rbx], rdx
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock inc qword ptr [r14 + rdx]
lfence
not rax
lfence
mov rsi, 5592
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
movsx rbx, byte ptr [r14 + rcx]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], rdi
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
