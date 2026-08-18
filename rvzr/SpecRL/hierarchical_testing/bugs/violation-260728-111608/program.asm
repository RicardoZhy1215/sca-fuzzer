.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or rbx, rcx
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
mov rax, rdi
paddq xmm2, xmm1
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rcx
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
xor rcx, rcx
and rbx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rbx]
setb cl
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx
pxor xmm5, xmm2
and rax, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rax]
mov rcx, rdx
setb cl
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx]
dec rcx
pextrq rcx, xmm7, 0
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx]
neg rcx
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdx
and rax, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rax]
setb cl
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rdx]
setb cl
setb dl
imul rcx, rsi
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rsi
pxor xmm0, xmm1
and rax, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rax]
mov rcx, rdi
cmp rcx, rbx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
mov rcx, rbx
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdi]
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
imul rbx, rax
neg rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
