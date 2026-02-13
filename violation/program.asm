.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 53 # instrumentation
sbb bl, 3 
or ax, 1 # instrumentation
and dx, ax # instrumentation
shr dx, 1 # instrumentation
div ax 
add bl, -77 # instrumentation
adc al, 20 
and rcx, 0b1111111111111 # instrumentation
sbb bx, word ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mul word ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
add si, word ptr [r14 + rcx] 
imul ax, ax 
cmp rax, 1650834729 
and rax, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rax] 
neg sil 
and rdx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdx], bl 
adc bx, -5 
add al, -5 
jbe .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
add bl, 22 # instrumentation
and rcx, 0b1111111111111 # instrumentation
adc rsi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmp cl, byte ptr [r14 + rdi] 
adc dl, al 
add ecx, 11 
sub rbx, rcx 
imul cl 
sbb cl, cl 
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], -26 
and rbx, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
