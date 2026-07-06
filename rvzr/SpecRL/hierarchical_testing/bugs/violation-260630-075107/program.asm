.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi
not rbx
pextrq rdi, xmm4, 0
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx
pextrq rcx, xmm1, 0
and rdx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx
and rdi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdi]
imul rsi, rsi
dec rdi
adc rsi, rdi
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
setl dil
jmp .bb_0.1
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2600
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi]
por xmm5, xmm1
adc rdx, rdi
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx
inc rsi
setl cl
and rdi, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdi]
mov rdx, rcx
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rax]
test rax, rbx
pmuludq xmm4, xmm4
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rdx
setb dil
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rbx
and rax, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rax
pcmpeqd xmm0, xmm4
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx
neg rax
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx
mov rdx, 200
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi
xor rcx, rsi
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi]
imul rdi, rcx
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
