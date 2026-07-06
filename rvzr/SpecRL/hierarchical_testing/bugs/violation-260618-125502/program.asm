.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setz dl
and rcx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rcx
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rcx
pextrq rdx, xmm2, 0
setb bl
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
or rsi, rdx
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdx]
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
not rdx
xor rdi, rdx
neg rdx
sbb rdi, rbx
xor rdx, rcx
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rax
xor rbx, rcx
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rcx
and rdi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdi]
dec rdx
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rcx
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
pextrq rdx, xmm2, 0
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rax
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rax]
cmp rax, rdx
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
paddq xmm5, xmm3
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
setz dl
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
