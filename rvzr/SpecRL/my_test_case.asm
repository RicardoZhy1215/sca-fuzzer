.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
cld  # instrumentation
sub di, di 
and rdx, 0b1111111111000 # instrumentation
lock bts dword ptr [r14 + rdx], 6 
and rcx, 0b1111111111111 # instrumentation
rol word ptr [r14 + rcx], cl 
stc  # instrumentation
sub dil, -80 
cmovz rbx, rsi 
and rdi, 0b1111111111111 # instrumentation
add rdi, r14 # instrumentation
scasw  
sub rdi, r14 # instrumentation
lea ecx, qword ptr [rsi + rdi + 7541] 
btr ecx, esi 
and rax, 0b1111111110000 # instrumentation
paddsb xmm1, xmmword ptr [r14 + rax] 
bts si, cx 
pextrb esi, xmm0, 202 
adc cl, 23 
movsx rax, cl 
setz al 
punpcklwd xmm0, xmm2 
and rbx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rbx], dil 
rcr ecx, cl 
stc  # instrumentation
and rdx, 0b1111111111111 # instrumentation
pmovzxdq xmm4, qword ptr [r14 + rdx] 
or eax, -163460800 
and rsi, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
rcr byte ptr [r14 + rbx], 48 
stc  # instrumentation
shr rcx, 36 
stc  # instrumentation
and rdx, 0b1111111110000 # instrumentation
punpckhdq xmm0, xmmword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
