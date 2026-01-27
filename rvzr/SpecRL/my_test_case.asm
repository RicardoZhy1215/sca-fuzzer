.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
or esi, 0b1000000000000000000000000000000 # instrumentation
bsf ecx, esi 
punpckldq xmm4, xmm5 
and rdi, 0b1111111111111 # instrumentation
popcnt rsi, qword ptr [r14 + rdi] 
shr rbx, 1 
stc  # instrumentation
neg dl 
sub al, bl 
and rbx, 0b1111111111111 # instrumentation
movddup xmm2, qword ptr [r14 + rbx] 
cmovnb rdx, rcx 
jb .bb_0.1 
jmp .exit_0 
.bb_0.1:
add al, 65 # instrumentation
cmovnz bx, cx 
cmovle edx, ebx 
and rsi, 0b1111111111111 # instrumentation
and dx, word ptr [r14 + rsi] 
and eax, 84 
shr al, cl 
stc  # instrumentation
and rcx, 0b1111111111111 # instrumentation
rcl dword ptr [r14 + rcx], 1 
stc  # instrumentation
sub eax, -202185420 
and rcx, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdx], bl 
and rdi, 0b1111111111111 # instrumentation
shld word ptr [r14 + rdi], cx, 77 
stc  # instrumentation
lea edx, qword ptr [rax + rdx] 
and rbx, 0b1111111111111 # instrumentation
shr byte ptr [r14 + rbx], cl 
stc  # instrumentation
and rdx, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bts word ptr [r14 + rdx], si 
and rsi, 0b1111111110000 # instrumentation
pmaxsb xmm3, xmmword ptr [r14 + rsi] 
adc ecx, -94 
or sil, sil 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
