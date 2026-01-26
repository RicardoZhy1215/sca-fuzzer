.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rax, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rax] 
verw dx 
and rax, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rax] 
pmovzxbd xmm6, xmm4 
lea ebx, qword ptr [rcx] 
cmovbe dx, bx 
or eax, 0b1000000000000000000000000000000 # instrumentation
bsr eax, eax 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], cl 
loope .bb_0.1 
jmp .exit_0 
.bb_0.1:
cld  # instrumentation
movq rsi, xmm5 
and rdi, 0b1111111111111 # instrumentation
add rdi, r14 # instrumentation
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repne insd  
sub rdi, r14 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rdi] 
movsx bx, cl 
sub esi, edi 
lea edi, qword ptr [rdx + rdx + 61683] 
and rcx, 0b1111111111111 # instrumentation
shl byte ptr [r14 + rcx], 135 
stc  # instrumentation
lea esi, qword ptr [rcx + rax + 43344] 
or al, 67 
out 212, al 
and rcx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rcx], dx 
and rcx, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
rcl word ptr [r14 + rdi], cl 
stc  # instrumentation
and rax, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rax], -81 
rol ax, 1 
stc  # instrumentation
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
