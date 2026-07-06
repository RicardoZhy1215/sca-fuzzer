.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pextrq rsi, xmm0, 0
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rcx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rax
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi
and rbx, rdx
pand xmm7, xmm5
pextrq rdx, xmm3, 0
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx]
test rcx, rbx
and rbx, rdx
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rcx]
or rcx, rbx
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rax
dec rax
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rax]
paddq xmm1, xmm5
and rdx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rdi]
pxor xmm1, xmm2
test rax, rbx
jmp .bb_0.1
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rdx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rdi
adc rdi, rbx
and rdx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rax]
setl dl
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdi
dec rax
adc rsi, rbx
xor rdx, rbx
cmp rcx, rbx
not rbx
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
