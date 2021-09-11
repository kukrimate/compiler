// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation
//

use super::ast::{Type,Expr,Init,Stmt,Vis,Func,File};
use std::collections::HashMap;
use std::rc::Rc;
use std::process::Command;
use tempfile::tempdir;

macro_rules! print_or_die {
    ($file:expr, $($args:expr),*) => {
        writeln!($file, $($args),*).unwrap()
    }
}

fn gen_static_init<T: std::io::Write>(file: &File, init: &Init, output: &mut T) {
    match init {
        Init::Base(expr) => {
            match expr {
                Expr::U8(v)  => { print_or_die!(output, "db {}", v); },
                Expr::I8(v)  => { print_or_die!(output, "db {}", v); },
                Expr::U16(v) => { print_or_die!(output, "dw {}", v); },
                Expr::I16(v) => { print_or_die!(output, "dw {}", v); },
                Expr::U32(v) => { print_or_die!(output, "dd {}", v); },
                Expr::I32(v) => { print_or_die!(output, "dd {}", v); },
                Expr::U64(v) => { print_or_die!(output, "dq {}", v); },
                Expr::I64(v) => { print_or_die!(output, "dq {}", v); },
                // Static can be initialized by another static
                Expr::Ident(ident) => {
                    if let Some(ref s) = file.statics.get(ident) {
                        // FIXME: fill with zeroes if the referenced static doesn't have an initializer
                        let ref refed_init = s.init.as_ref().unwrap();
                        gen_static_init(file, refed_init, output);
                    } else {
                        panic!("Unknown")
                    }
                },
                // Or by a pointer to a static (or string literal)
                Expr::Ref(dest) => {
                    match &**dest {
                        Expr::Ident(ident) => {
                            print_or_die!(output, "dq {}", ident);
                        },
                        _ => panic!("Non constant static initializer"),
                    }
                },
                _ => panic!("Non constant static initializer"),
            }
        },
        Init::List(ref list) => for i in list {
            gen_static_init(file, i, output);
        },
    };
}

fn assert_integral(dtype: &Type) {
    match dtype {
        Type::VOID => panic!("Math on non-integral type"),
        Type::Array {..} => panic!("Math on non-integral type"),
        Type::Record {..} => panic!("Math on non-integral type"),
        _ => (),
    }
}

// Registers usable as temporary values
// Rip, Rsp, Rbp are reserved for the compiler's use
#[derive(Clone, Copy, Debug)]
enum Reg {
    Rax = (1 << 0),
    Rbx = (1 << 1),
    Rcx = (1 << 2),
    Rdx = (1 << 3),
    Rsi = (1 << 4),
    Rdi = (1 << 5),
    R8  = (1 << 6),
    R9  = (1 << 7),
    R10 = (1 << 8),
    R11 = (1 << 9),
    R12 = (1 << 10),
    R13 = (1 << 11),
    R14 = (1 << 12),
    R15 = (1 << 13),
}

impl Reg {
    fn to_str(&self) -> &str {
        match self {
            Reg::Rax => "rax", Reg::Rbx => "rbx", Reg::Rcx => "rcx",
            Reg::Rdx => "rdx", Reg::Rsi => "rsi", Reg::Rdi => "rdi",
            Reg::R8  => "r8" , Reg::R9  => "r9" , Reg::R10 => "r10",
            Reg::R11 => "r11", Reg::R12 => "r12", Reg::R13 => "r13",
            Reg::R14 => "r14", Reg::R15 => "r15"
        }
    }
}

// List of all registers (in allocation order, callee-saved ones come first)
static ALL_REGS: [Reg; 14] = [
    Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15, // Callee saved
    Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi, // Caller saved
    Reg::R8,  Reg::R9,  Reg::R10, Reg::R11,
];

// List of caller saved registers
static CALLER_SAVED: [Reg; 9] = [
    Reg::Rax, Reg::Rcx, Reg::Rdx, Reg::Rsi, Reg::Rdi,
    Reg::R8,  Reg::R9,  Reg::R10, Reg::R11,
];

// List of registers used for parameters
static PARAM_REGS: [Reg; 6] = [
    Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9
];

// Free register mask
struct RegMask {
    usedregs: u16
}

impl RegMask {
    fn new() -> RegMask {
        RegMask {
            usedregs: 0
        }
    }

    fn alloc_reg(&mut self) -> Option<Reg> {
        for reg in ALL_REGS {
            let mask = reg as u16;
            if self.usedregs & mask == 0 {
                self.usedregs |= mask;
                return Some(reg);
            }
        }
        None
    }

    fn clear_reg(&mut self, reg: Reg) {
        let mask = reg as u16;
        if self.usedregs & mask == 0 {
            panic!("Register {:?} is already clear", reg);
        }
        self.usedregs &= !mask;
    }
}
struct FuncCtx {
    // Register allocator
    regmask: RegMask,
    // Local variables
    locals: HashMap<Rc<str>, (Type, usize)>,
    // Stack frame size
    frame_size: usize,
}

impl FuncCtx {
    // Find the effective address of a local variable
    fn local_ea(&self, name: &Rc<str>) -> String {
        let (_, offset) = self.locals.get(name).unwrap();
        format!("[ebp - {}]", self.frame_size - offset)
    }
}

enum Val {
    // Immediate integer value
    Immed(usize),
    // Integer (or pointer) value in a register
    Reg(Reg),
    // Value is stored on the stack
    Stack(usize),
    // Value is stored in static storage
    Static(Rc<str>, usize),
    // Dereference of a pointer
    Deref(Box<Val>, usize),
}

// Generate pointer creation code
fn gen_ref<T: std::io::Write>(fctx: &mut FuncCtx, val: Val, output: &mut T) -> Val {
    match val {
        Val::Stack(offset) => {
            // FIXME: temporaries won't necessarily fit into registers
            let reg = fctx.regmask.alloc_reg().unwrap();
            print_or_die!(output, "lea {}, [ebp - {}]",
                                    reg.to_str(), fctx.frame_size - offset);
            Val::Reg(reg)
        },
        Val::Static(name, offset) => {
            let reg = fctx.regmask.alloc_reg().unwrap();
            print_or_die!(output, "lea {}, [{} + {}]",
                                    reg.to_str(), name, offset);
            Val::Reg(reg)

        },
        _ => panic!("Cannot take address of rvalue!"),
    }
}

// Create value at an offset from an lvalue
fn add_offs(val: Val, add_offset: usize) -> Val {
    match val {
        Val::Stack(offset) => Val::Stack(offset + add_offset),
        Val::Static(name, offset) => Val::Static(name, offset + add_offset),
        Val::Deref(derefed_val, offset) => Val::Deref(derefed_val, offset + add_offset),
        _ => panic!("Cannot take offset from rvalue!"),
    }
}

// Save all caller-saved registers before a call
fn pushcallersaved<T: std::io::Write>(usedregs: u16, output: &mut T) -> (u16, bool) {
    let mut savemask = 0u16;
    let mut pad = false;
    for reg in CALLER_SAVED {
        let regmask = reg as u16;
        if usedregs & regmask != 0 {
            savemask |= regmask;
            pad = !pad;
            print_or_die!(output, "push {}", reg.to_str());
        }
    }
    if pad {
        // Make sure stack alingment is preserved
        print_or_die!(output, "push 0");
    }
    (savemask, pad)
}

// Restore saved caller-saved registers
fn popcallersaved<T: std::io::Write>(savemask: u16, pad: bool, output: &mut T) {
    if pad {
        // Remove padding if present
        print_or_die!(output, "add rsp, 8");
    }

    for reg in CALLER_SAVED.iter().rev() {
        let regmask = *reg as u16;
        if savemask & regmask != 0 {
            print_or_die!(output, "pop {}", reg.to_str());
        }
    }
}


fn gen_expr<T: std::io::Write>(
        file: &File,
        fctx: &mut FuncCtx,
        in_expr: &Expr,
        output: &mut T) -> (Type, Val) {

    match in_expr {
        Expr::U8(v)  => (Type::U8, Val::Immed(*v as usize)),
        Expr::I8(v)  => (Type::I8, Val::Immed(*v as usize)),
        Expr::U16(v) => (Type::U16, Val::Immed(*v as usize)),
        Expr::I16(v) => (Type::I16, Val::Immed(*v as usize)),
        Expr::U32(v) => (Type::U32, Val::Immed(*v as usize)),
        Expr::I32(v) => (Type::I32, Val::Immed(*v as usize)),
        Expr::U64(v) => (Type::U64, Val::Immed(*v as usize)),
        Expr::I64(v) => (Type::I64, Val::Immed(*v as usize)),

        Expr::Ident(ident) => {
            if let Some((dtype, offset)) = fctx.locals.get(ident) {
                (dtype.clone(), Val::Stack(*offset))
            } else if let Some(s) = file.statics.get(ident) {
                (s.dtype.clone(), Val::Static(ident.clone(), 0))
            } else {
                panic!("Unknown identifier {}", ident);
            }
        },

        Expr::Ref(expr) => {
            let (src_type, src_val) = gen_expr(file, fctx, expr, output);
            (Type::Ptr { base_type: Box::from(src_type.clone()) },
                gen_ref(fctx, src_val, output))
        },

        Expr::Deref(expr) => {
            let (src_type, src_val) = gen_expr(file, fctx, expr, output);
            match src_type {
                // Create dereferenced value from pointer
                Type::Ptr { base_type } =>
                    ((*base_type).clone(), Val::Deref(Box::from(src_val), 0)),
                // Create new value with the array's element type at the same location
                // (offset was already added to the array's value if this deref was
                // de-sugared from array indexing)
                Type::Array { elem_type, .. } =>
                    ((*elem_type).clone(), src_val),
                _ => panic!("Can't dereference non-reference type"),
            }
        }

        Expr::Field(expr, ident) => {
            let (src_type, src_val) = gen_expr(file, fctx, expr, output);
            match src_type {
                Type::Record(record) => {
                    if let Some((field_type, field_offset)) = record.fields.get(ident) {
                        (field_type.clone(), add_offs(src_val, *field_offset))
                    } else {
                        panic!("Non-existent field {} accessed", ident);
                    }
                },
                _ => panic!("Can't access field of non-record type"),
            }
        },

        Expr::Call(expr, params) => {
            // Find target function to call
            let func = match &**expr {
                Expr::Ident(ident) => {
                    if let Some(func) = file.funcs.get(ident) {
                        func
                    } else {
                        panic!("Call to undefined function {}", ident);
                    }
                },
                _ => panic!("Function name must be an identifier"),
            };
            // Save callee saved registers
            let (savemask, pad) = pushcallersaved(fctx.regmask.usedregs, output);

            for (i, param) in params.iter().enumerate() {
                // FIXME: support more than 6 parameters
                let param_reg = PARAM_REGS[i];
                let (param_type, param_val) = gen_expr(file, fctx, param, output);
            }
            if func.varargs {
                // NOTE: Varargs functions need to be told the number of
                // floating-point arguments
                print_or_die!(output, "xor rax, rax");
            }
            print_or_die!(output, "call {}", func.name);

            // Restore registers
            popcallersaved(savemask, pad, output);

            // FIXME: rax might be overwritten above
            (func.rettype.clone(), Val::Reg(Reg::Rax))
        },

/*
        Expr::Inv(ref expr) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, frame_size, locals, output);
            assert_integral(&src_type);
            val_to_accum(frame_size, accum, &src_val, output);
            print_or_die!(output, "not {}", accum);
            (src_type, Val::Accum)
        },

        Expr::Neg(ref expr) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, frame_size, locals, output);
            assert_integral(&src_type);
            val_to_accum(frame_size, accum, &src_val, output);
            print_or_die!(output, "neg {}", accum);
            (src_type, Val::Accum)
        },

        Expr::Add(ref lhs, ref rhs) => {
            // Evaluate LHS first
            let (lhs_type, lhs_val) = gen_expr(file, accum, lhs, frame_size, locals, output);
            assert_integral(&lhs_type);
            val_to_accum(frame_size, accum, &lhs_val, output);
            // Save LHS value to stack (twice for alingment)
            print_or_die!(output, "push {}", accum);
            print_or_die!(output, "push {}", accum);

            // Now we can eval the RHS
            let (rhs_type, rhs_val) = gen_expr(file, accum, rhs, frame_size, locals, output);
            assert_integral(&rhs_type);
            val_to_accum(frame_size, accum, &rhs_val, output);

            // Add RHS to LHS
            print_or_die!(output, "add {}, [rsp]", accum);
            print_or_die!(output, "add rsp, 16");
            (lhs_type, Val::Accum)
        },

        Expr::Sub(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },
        Expr::Mul(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },
        Expr::Div(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },
        Expr::Rem(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },
        Expr::Or(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },
        Expr::And(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },
        Expr::Xor(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },
        Expr::Lsh(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },
        Expr::Rsh(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },

        Expr::Cast(ref expr, dtype) => {
            gen_expr(expr, locals, data);
            dtype.clone()
        },*/
        _ => todo!("expression {:?}", in_expr),
    }
}

fn gen_func<T: std::io::Write>(file: &File, func: &Func, output: &mut T) {
    print_or_die!(output, "{}:", func.name);

    // Create context
    let mut ctx = FuncCtx {
        regmask: RegMask::new(),
        locals: HashMap::new(),
        frame_size: 0,
    };

    // Allocate stack slots for parameters
    // We note than the param to offset map for later use
    let mut params = Vec::new();
    for (i, (name, t)) in func.params.iter().enumerate() {
        if t.get_size() > 8 {
            panic!("FIXME: larger-than register parameters!");
        }
        // Add local variable for parameter
        ctx.locals.insert(name.clone(), (t.clone(), ctx.frame_size));
        // Save index to stack offset map
        params.push((i, ctx.frame_size));
        // Increase stack frame size
        // HACK!: assume all parameters are 8 bytes for simpler moves
        ctx.frame_size += 8;
    }

    // Allocate stack slots for auto variables
    for stmt in &func.stmts {
        match stmt {
            Stmt::Auto(name, dtype, _) => {
                ctx.locals.insert(name.clone(), (dtype.clone(), ctx.frame_size));
                ctx.frame_size += dtype.get_size();
            },
            _ => (),
        }
    }

    // Align frame size to a multiple of 16, then add 8 (sysv abi)
    ctx.frame_size = (ctx.frame_size + 15) / 16 * 16 + 8;
    // Generate function prologue
    print_or_die!(output, "push rbp\nmov rbp, rsp\nsub rsp, {}", ctx.frame_size);

    // Move parameters from registers to stack
    for (i, offset) in params {
        // FIXME: support more than 6 parameters
        print_or_die!(output, "mov qword [rbp - {}], {}",
                ctx.frame_size - offset,
                PARAM_REGS[i].to_str());
    }

    for stmt in &func.stmts {
        match stmt {
            Stmt::Label(label) => {
                print_or_die!(output, ".{}:", label);
            },
            /*Stmt::Auto(ident, dtype, init) => {},
            Stmt::Set(ref dest, ref expr) => {
                let (ltype, lval) = gen_expr(file, "r10", dest, frame_size, &locals, output);
                let (rtype, rval) = gen_expr(file, "rax", expr, frame_size, &locals, output);
                val_to_accum(frame_size, "rax", &rval, output);
                store_accum_to_val("rax", &lval, output);
            }*/
            Stmt::Eval(ref expr) => {
                gen_expr(file, &mut ctx, expr, output);
            },
            Stmt::Jmp(label) => {
                print_or_die!(output, "jmp .{}", label);
            },
            Stmt::Ret(_) => {
                print_or_die!(output, "jmp done");
            },
            _ => todo!("statement {:?}", stmt),
        }
    }

    // Generate function epilogue
    print_or_die!(output, "done:\nleave\nret");
}

pub fn gen_asm<T: std::io::Write>(input: &File, output: &mut T) {
    let mut exports = Vec::new();
    let mut externs = Vec::new();

    let mut bss = HashMap::new();

    // Generate data
    print_or_die!(output, "section .data");
    for (_, cur_static) in &input.statics {
        match cur_static.vis {
            Vis::Export => exports.push(cur_static.name.clone()),
            Vis::Extern => externs.push(cur_static.name.clone()),
            _ => (),
        };

        match &cur_static.init {
            // Generate static initializer in .data
            Some(ref init) => {
                print_or_die!(output, "{}:", cur_static.name);
                gen_static_init(&input, init, output);
            },
            // Take note for bss allocation later
            None => {
                bss.insert(cur_static.name.clone(),
                    cur_static.dtype.get_size());
            },
        };
    }

    // Generate bss
    print_or_die!(output, "section .bss");
    for (name, len) in bss {
        print_or_die!(output, "{}: resb {}", name, len);
    }

    // Generate functions
    print_or_die!(output, "section .text");

    for (_, func) in &input.funcs {
        match func.vis {
            Vis::Export => exports.push(func.name.clone()),
            Vis::Extern => externs.push(func.name.clone()),
            _ => (),
        };
        // Generate body for non-extern function
        if func.vis != Vis::Extern {
            gen_func(&input, &func, output);
        }
    }

    // Generate markers
    for exp in exports {
        print_or_die!(output, "global {}", exp);
    }
    for ext in externs {
        print_or_die!(output, "extern {}", ext);
    }
}

pub fn gen_obj<T: std::io::Write>(input: &File, output: &mut T) {
    let tdir = tempdir().unwrap();
    let tasm = tdir.path().join("asm");
    let tobj = tdir.path().join("obj");

    // Generate assembly
    gen_asm(input, &mut std::fs::File::create(&tasm).unwrap());

    // Assemble
    let status = Command::new("nasm")
        .args(["-f", "elf64", "-o", tobj.to_str().unwrap(),
                tasm.to_str().unwrap()])
        .status()
        .expect("failed to run nasm");

    if !status.success() {
        panic!("assembly with nasm failed");
    }

    // Write assembly result to output
    output.write(&std::fs::read(tobj).unwrap()).unwrap();
}
