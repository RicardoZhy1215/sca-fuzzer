.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 86 # instrumentation
cmp rdx, rdi
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], 5
mov rsi, rdi
add rax, rbx
add rdi, 5
cmp rdx, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
js .bb_0.1
jmp .exit_0
.bb_0.1:
xor rcx, rsi
mov rsi, rbx
xor rax, rbx
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rdx
mov rdx, rcx
cmp rax, rbx
and rbx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rbx]
xor rbx, rbx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
xor rax, rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
