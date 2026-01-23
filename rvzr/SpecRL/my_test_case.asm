.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
cld  # instrumentation
lea esi, qword ptr [rbx + rax] 
lea rbx, qword ptr [rsi + rcx + 64085] 
and rdi, 0b1111111111111 # instrumentation
add rdi, r14 # instrumentation
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe stosb  
sub rdi, r14 # instrumentation
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], -82 
and rbx, 0b1111111110000 # instrumentation
paddq xmm7, xmmword ptr [r14 + rbx] 
pmovzxbq xmm7, xmm2 
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdx], -21 
and rdi, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rdi] 
cmp al, 127 
packuswb xmm1, xmm6 
and rsi, 0b1111111110000 # instrumentation
psllq xmm4, xmmword ptr [r14 + rsi] 
and rax, 0b1111111110000 # instrumentation
paddusb xmm5, xmmword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdx], 19 
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], 52 
and rax, 0b1111111110000 # instrumentation
pavgb xmm0, xmmword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
movzx ebx, byte ptr [r14 + rax] 
and rdi, 0b1111111110000 # instrumentation
psubsb xmm3, xmmword ptr [r14 + rdi] 
bswap rdx 
sub ax, -12330 
and rcx, 0b1111111110000 # instrumentation
packusdw xmm5, xmmword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
cvttss2si rcx, dword ptr [r14 + rsi] 
sar si, 1 
stc  # instrumentation
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
