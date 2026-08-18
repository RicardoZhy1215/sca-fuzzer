.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
por xmm0, xmm5
lea rbx, qword ptr [rdi + rbx + 1]
dec rcx
pextrq rdx, xmm3, 0
neg rcx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rcx
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rbx]
inc rdx
setb sil
xor rbx, rdi
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rsi
xor rbx, rbx
and rdi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdi]
sbb rcx, rdx
setz dl
neg rsi
setl al
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx]
pextrq rdx, xmm7, 0
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx
inc rdx
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rcx
and rbx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rax]
mov rbx, 4168
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
