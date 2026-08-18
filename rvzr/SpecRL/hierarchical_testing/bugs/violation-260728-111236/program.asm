.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pextrq rdi, xmm2, 0
and rbx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rsi]
imul rbx, rsi
pxor xmm4, xmm0
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
setb sil
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
setz sil
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rcx
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
pxor xmm3, xmm6
inc rdi
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
and rbx, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rbx]
inc rdi
and rdx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdx]
pxor xmm6, xmm5
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rbx
neg rcx
not rcx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rcx
movd xmm4, esi
movq xmm0, rax
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdx]
imul rcx, rcx
and rdi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax
and rax, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rsi
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx]
cmp rbx, rdx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi
neg rbx
neg rsi
and rdi, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rax]
imul rbx, rdx
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
adc rdi, rdi
por xmm6, xmm2
paddq xmm2, xmm7
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
