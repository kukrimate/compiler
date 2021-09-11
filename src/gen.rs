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

// Registers usable as temporary values
// Rip, Rsp, Rbp are reserved for the compiler's use
#[derive(Clone, Copy, Debug, PartialEq)]
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

    fn maybe_alloc_specific_reg(&mut self, want: Reg) -> Option<Reg> {
        let wantmask = want as u16;
        if self.usedregs & wantmask == 0 {
            self.usedregs |= wantmask;
            Some(want)
        } else {
            self.alloc_reg()
        }
    }

    fn really_alloc_specific_reg(&mut self, want: Reg) -> Option<Reg> {
        let wantmask = want as u16;
        if self.usedregs & wantmask == 0 {
            self.usedregs |= wantmask;
            Some(want)
        } else {
            None
        }
    }

    fn clear_reg(&mut self, reg: Reg) {
        let mask = reg as u16;
        if self.usedregs & mask == 0 {
            panic!("Register {:?} is already clear", reg);
        }
        self.usedregs &= !mask;
    }

    // Save all caller-saved registers before a call (except the one used for the return value)
    fn pushcallersaved<T: std::io::Write>(&mut self, ret: Reg, output: &mut T) -> (u16, bool) {
        let mut savemask = 0u16;
        let mut pad = false;

        // Save all caller-saved regsiters marked as used
        for reg in CALLER_SAVED {
            let regmask = reg as u16;
            if reg != ret && self.usedregs & regmask != 0 {
                savemask |= regmask;
                pad = !pad;
                print_or_die!(output, "push {}", reg.to_str());
            }
        }
        // Make sure stack alingment is preserved
        if pad {
            print_or_die!(output, "push 0");
        }
        // Clear the caller-saved registers in the use mask
        self.usedregs &= !savemask;

        (savemask, pad)
    }

    // Restore saved caller-saved registers
    fn popcallersaved<T: std::io::Write>(&mut self, savemask: u16, pad: bool, output: &mut T) {
        // Remove padding if present
        if pad {
            print_or_die!(output, "add rsp, 8");
        }

        // Make sure we didn't acccidently clobber one of the saved registers
        if self.usedregs & savemask != 0 {
            panic!("Saved register clobbered in call");
        }

        // Restore all saved registers
        for reg in CALLER_SAVED.iter().rev() {
            let regmask = *reg as u16;
            if savemask & regmask != 0 {
                print_or_die!(output, "pop {}", reg.to_str());
            }
        }

        // Set restored registers to clobbered state again
        self.usedregs |= savemask;
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

enum RVal {
    // Immediate integer value
    Immed(usize),
    // Integer (or pointer) value in a register
    Reg(Reg),
}

impl RVal {
    fn to_string(&self) -> String {
        match self {
            RVal::Immed(val) => format!("{}", val),
            RVal::Reg(reg) => String::from(reg.to_str()),
        }
    }
}

enum Val {
    // Unaddressable temporary value
    RVal(RVal),
    // Value is stored on the stack
    Stack(usize),
    // Value is stored in static storage
    Static(Rc<str>, usize),
    // Dereference of a pointer
    Deref(Box<Val>, usize),
}

fn rval_src(rval: RVal) -> (String, Option<Reg>) {
    match rval {
        RVal::Immed(val) => {
            (format!("{}", val), None)
        },
        RVal::Reg(reg) => {
            (String::from(reg.to_str()), Some(reg))
        },
    }
}

// Make rvalue from a value
fn make_rval<T: std::io::Write>(fctx: &mut FuncCtx, val: Val, output: &mut T) -> RVal {
    match val {
        Val::RVal(rval) => rval,
        Val::Stack(offset) => {
            let dreg = fctx.regmask.alloc_reg().unwrap();
            print_or_die!(output, "mov {}, [rbp - {}]",
                                    dreg.to_str(), fctx.frame_size - offset);
            RVal::Reg(dreg)
        },
        Val::Static(name, offset) => {
            let dreg = fctx.regmask.alloc_reg().unwrap();
            print_or_die!(output, "mov {}, [{} + {}]",
                                    dreg.to_str(), name, offset);
            RVal::Reg(dreg)
        },
        Val::Deref(ptr_val, offset) => {
            // Compute the value of the derefed pointer as an rvalue
            let ptr_rval = make_rval(fctx, *ptr_val, output);
            // If it was an immediate than we must allocate a register
            let (rval_src, rval_reg) = rval_src(ptr_rval);
            let dreg = if let Some(dreg) = rval_reg {
                dreg
            } else {
                fctx.regmask.alloc_reg().unwrap()
            };
            // Now we can perform the dereference
            print_or_die!(output, "mov {}, [{} + {}]",
                                    dreg.to_str(), rval_src, offset);
            RVal::Reg(dreg)
        },
    }
}

// Make an rvalue pointer to a value
fn make_ptr<T: std::io::Write>(fctx: &mut FuncCtx, val: Val, output: &mut T) -> RVal {
    match val {
        Val::Stack(offset) => {
            let dreg = fctx.regmask.alloc_reg().unwrap();
            print_or_die!(output, "lea {}, [rbp - {}]",
                                    dreg.to_str(), fctx.frame_size - offset);
            RVal::Reg(dreg)
        },
        Val::Static(name, offset) => {
            let dreg = fctx.regmask.alloc_reg().unwrap();
            print_or_die!(output, "lea {}, [{} + {}]",
                                    dreg.to_str(), name, offset);
            RVal::Reg(dreg)
        },
        Val::Deref(ptr_val, offset) => {
            // Compute the value of the derefed pointer as an rvalue
            let ptr_rval = make_rval(fctx, *ptr_val, output);
            // If it was an immediate than we must allocate a register
            let (rval_src, rval_reg) = rval_src(ptr_rval);
            let dreg = if let Some(dreg) = rval_reg {
                dreg
            } else {
                fctx.regmask.alloc_reg().unwrap()
            };
            // Now we can perform the dereference
            print_or_die!(output, "lea {}, [{} + {}]",
                                    dreg.to_str(), rval_src, offset);
            RVal::Reg(dreg)
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

fn gen_expr<T: std::io::Write>(
        file: &File,
        fctx: &mut FuncCtx,
        in_expr: &Expr,
        output: &mut T) -> (Type, Val) {

    match in_expr {
        Expr::U8(v)  => (Type::U8, Val::RVal(RVal::Immed(*v as usize))),
        Expr::I8(v)  => (Type::I8, Val::RVal(RVal::Immed(*v as usize))),
        Expr::U16(v) => (Type::U16, Val::RVal(RVal::Immed(*v as usize))),
        Expr::I16(v) => (Type::I16, Val::RVal(RVal::Immed(*v as usize))),
        Expr::U32(v) => (Type::U32, Val::RVal(RVal::Immed(*v as usize))),
        Expr::I32(v) => (Type::I32, Val::RVal(RVal::Immed(*v as usize))),
        Expr::U64(v) => (Type::U64, Val::RVal(RVal::Immed(*v as usize))),
        Expr::I64(v) => (Type::I64, Val::RVal(RVal::Immed(*v as usize))),

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
                Val::RVal(make_ptr(fctx, src_val, output)))
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

            // Allocate destination for the returned value
            // NOTE: we prefer to allocate rax here to avoid extra moves
            // FIXME: might not be able to allocate a register
            let dreg = fctx.regmask.maybe_alloc_specific_reg(Reg::Rax).unwrap();

            // Save caller-saved registers
            let (savemask, pad) = fctx.regmask.pushcallersaved(dreg, output);

            // Evaluate parameter expressions
            for (i, param) in params.iter().enumerate() {
                let param_reg = fctx.regmask.
                    really_alloc_specific_reg(PARAM_REGS[i]).unwrap();

                let (_, param_val) = gen_expr(file, fctx, param, output);
                let param_rval = make_rval(fctx, param_val, output);

                match param_rval {
                    RVal::Immed(val) => {
                        print_or_die!(output, "mov {}, {}",
                            param_reg.to_str(), val);
                    },
                    RVal::Reg(reg) => {
                        if param_reg != reg {
                            print_or_die!(output, "mov {}, {}",
                                param_reg.to_str(), reg.to_str());
                            fctx.regmask.clear_reg(reg);
                        }
                    },
                }
            }

            if func.varargs {
                // Provide number of floating arguments for varargs functions
                print_or_die!(output, "xor rax, rax");
            }
            // Call function
            print_or_die!(output, "call {}", func.name);

            // Move return value to result register
            if dreg != Reg::Rax {
                print_or_die!(output, "mov {}, rax", dreg.to_str());
                fctx.regmask.clear_reg(Reg::Rax);
            }
            // Clear param clobbers
            for (i, _) in params.iter().enumerate() {
                fctx.regmask.clear_reg(PARAM_REGS[i]);
            }

            // Restore caller-saved registers
            fctx.regmask.popcallersaved(savemask, pad, output);

            (func.rettype.clone(), Val::RVal(RVal::Reg(dreg)))
        },

        Expr::Inv(expr) => {
            let (src_type, src_val) = gen_expr(file, fctx, expr, output);
            match make_rval(fctx, src_val, output) {
                RVal::Immed(val) => {
                    // Fold bitwise inverse of constant
                    (src_type, Val::RVal(RVal::Immed(!val)))
                },
                RVal::Reg(reg) => {
                    print_or_die!(output, "not {}", reg.to_str());
                    (src_type, Val::RVal(RVal::Reg(reg)))
                },
            }
        },

        Expr::Neg(expr) => {
            let (src_type, src_val) = gen_expr(file, fctx, expr, output);
            match make_rval(fctx, src_val, output) {
                RVal::Immed(val) => {
                    // Fold two's complement negation of constant
                    (src_type, Val::RVal(RVal::Immed(!val + 1)))
                },
                RVal::Reg(reg) => {
                    print_or_die!(output, "neg {}", reg.to_str());
                    (src_type, Val::RVal(RVal::Reg(reg)))
                },
            }
        },

        Expr::Add(lhs, rhs) => {
            let (lhs_type, lhs_val) = gen_expr(file, fctx, lhs, output);
            let (rhs_type, rhs_val) = gen_expr(file, fctx, rhs, output);
            if lhs_type != rhs_type {
                panic!("Cannot add different type expressions!");
            }

            let lhs_rval = make_rval(fctx, lhs_val, output);
            let rhs_rval = make_rval(fctx, rhs_val, output);

            match lhs_rval {
                RVal::Immed(l) => {
                    match rhs_rval {
                        RVal::Immed(r) => {
                            // Fold add of two constants
                            (lhs_type, Val::RVal(RVal::Immed(l + r)))
                        },
                        RVal::Reg(rreg) => {
                            print_or_die!(output, "add {}, {}", rreg.to_str(), l);
                            (lhs_type, Val::RVal(RVal::Reg(rreg)))
                        },
                    }
                },
                RVal::Reg(lreg) => {
                    match rhs_rval {
                        RVal::Immed(r) => {
                            print_or_die!(output, "add {}, {}", lreg.to_str(), r);
                            (lhs_type, Val::RVal(RVal::Reg(lreg)))
                        },
                        RVal::Reg(rreg) => {
                            assert!(lreg != rreg); // Shouldn't ever happen
                            print_or_die!(output, "add {}, {}",
                                lreg.to_str(), rreg.to_str());
                            fctx.regmask.clear_reg(rreg);
                            (lhs_type, Val::RVal(RVal::Reg(lreg)))
                        },
                    }
                },
            }

        },

/*
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

fn gen_set<T: std::io::Write>(fctx: &mut FuncCtx, dest: Val, src: RVal, output: &mut T) {
    match dest {
        Val::Stack(offset) => {
            print_or_die!(output, "mov [rbp - {}], {}",
                                    fctx.frame_size - offset, src.to_string());
        },
        Val::Static(name, offset) => {
            print_or_die!(output, "mov [{} + {}], {}",
                                    name, offset, src.to_string());
        },
        Val::Deref(derefed_val, offset) => {
            panic!("FIXME: deref set");
        },
        _ => panic!("Write to rvalue!"),
    }
}

fn gen_eval(fctx: &mut FuncCtx, val: Val) {
    match val {
        Val::RVal(rval) => match rval {
            RVal::Reg(reg) => fctx.regmask.clear_reg(reg),
            _ => (),
        },
        Val::Deref(ptr_val, _) => gen_eval(fctx, *ptr_val),
        _ => (),
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
            Stmt::Auto(ident, dtype, init) => {},
            Stmt::Set(dest, expr) => {
                let (dest_type, dest) = gen_expr(file, &mut ctx, dest, output);
                let (src_type, src) = gen_expr(file, &mut ctx, expr, output);
                if dest_type != src_type {
                    panic!("Set statement types differ, left: {:?}, right: {:?}", dest_type, src_type);
                }
                let src_rval = make_rval(&mut ctx, src, output);
                gen_set(&mut ctx,
                            dest,
                            src_rval,
                            output);
            }
            Stmt::Eval(ref expr) => {
                let (_, val) = gen_expr(file, &mut ctx, expr, output);
                gen_eval(&mut ctx, val);
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
