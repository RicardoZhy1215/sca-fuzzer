.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
cmp rdx, rax
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdi
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rax]
not rax
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx
test rbx, rcx
setb dl
test rbx, rdx
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rdi
and rdi, rsi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdx
and rbx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx
and rdi, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rdi]
setl dl
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax
or rdi, rdx
setnz dl
setl dl
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rsi]
test rdx, rdx
setnz cl
test rdx, rax
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi
and rbx, rdi
and rdi, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rbx
imul rcx, rbx
setb dl
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rdx
and rax, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rax]
not rdx
test rax, rdi
cmp rdx, rbx
test rcx, rbx
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
dec rdi
and rax, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
