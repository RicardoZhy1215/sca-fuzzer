.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pand xmm1, xmm2
and rdx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdx]
movd xmm0, eax
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rbx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdi
mov rdx, 4376
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx
inc rbx
sbb rdx, rcx
and rsi, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rsi]
dec rbx
pextrq rax, xmm2, 0
mov rsi, rdx
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rcx
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rdx]
setb cl
inc rbx
adc rbx, rax
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rax
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
setb bl
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
movd xmm2, esi
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
setb dl
psubq xmm5, xmm1
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax]
psubq xmm4, xmm7
xor rcx, rsi
dec rdx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rax
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx
or rsi, rsi
lea rcx, qword ptr [rcx + rcx + 1]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
