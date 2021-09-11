// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation
//

use super::ast::{Type,Expr,Init,Stmt,Vis,Func,File};
use std::collections::HashMap;
use std::rc::Rc;

macro_rules! print_or_die {
    ($file:expr, $($args:expr),*) => {
        writeln!($file, $($args),*).unwrap();
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

enum Val {
    // Accumulator contains the value itself
    Accum,
    // Accumulator contains a pointer to the value
    AccumRef,
    // Value is stored on the stack
    Stack(usize),
    // Value is stored in static storage
    Static(Rc<str>, usize),
}

// Load a reference to a value into the accumulator
// Aka for making a pointer to a value
fn ref_to_accum<T: std::io::Write>(frame_size: usize, accum: &str, val: &Val, output: &mut T) {
    match val {
        Val::Accum => panic!("Rvalue used as lvalue"),
        Val::AccumRef => (), // Already have the address (only ends up here when derefing a pointer we just ref'd)
        Val::Stack(offset) => { print_or_die!(output, "lea {}, [rbp - {}]", accum, frame_size - offset); },
        Val::Static(ident, offset) => { print_or_die!(output, "lea {}, [{} + {}]", accum, ident, offset); },
    }
}

// Load a value into the accumulator
fn val_to_accum<T: std::io::Write>(frame_size: usize, accum: &str, val: &Val, output: &mut T) {
    match val {
        Val::Accum => (), // Already have the value
        Val::AccumRef => { print_or_die!(output, "mov {}, [{}]", accum, accum); }, // Load value from pointer
        Val::Stack(offset) => { print_or_die!(output, "mov {}, [rbp - {}]", accum, frame_size - offset); }, // Load value from stack
        Val::Static(ident, offset) => { print_or_die!(output, "mov {}, [{} + {}]", accum, ident, offset); }, // Load value from static
    }
}

// Make a new value that refers to an offset from a previous value
fn field_val<T: std::io::Write>(accum: &str, val: &Val, field_offs: usize, output: &mut T) -> Val {
    match val {
        Val::Accum => panic!("Rvalue used as lvalue"),
        Val::AccumRef => { print_or_die!(output, "add {}, {}", accum, field_offs); Val::AccumRef },
        Val::Stack(offset) => Val::Stack(offset + field_offs),
        Val::Static(ident, offset) => Val::Static(ident.clone(), offset + field_offs),
    }
}

fn gen_expr<T: std::io::Write>(
        file: &File,
        accum: &str,
        in_expr: &Expr,
        frame_size: usize,
        locals: &HashMap<Rc<str>, (&Type, usize)>,
        output: &mut T) -> (Type, Val) {

    match in_expr {
        Expr::U8(v)  => { print_or_die!(output, "mov {}, {}", accum, v); (Type::U8, Val::Accum) },
        Expr::I8(v)  => { print_or_die!(output, "mov {}, {}", accum, v); (Type::I8, Val::Accum) },
        Expr::U16(v) => { print_or_die!(output, "mov {}, {}", accum, v); (Type::U16, Val::Accum) },
        Expr::I16(v) => { print_or_die!(output, "mov {}, {}", accum, v); (Type::I16, Val::Accum) },
        Expr::U32(v) => { print_or_die!(output, "mov {}, {}", accum, v); (Type::U32, Val::Accum) },
        Expr::I32(v) => { print_or_die!(output, "mov {}, {}", accum, v); (Type::I32, Val::Accum) },
        Expr::U64(v) => { print_or_die!(output, "mov {}, {}", accum, v); (Type::U64, Val::Accum) },
        Expr::I64(v) => { print_or_die!(output, "mov {}, {}", accum, v); (Type::I64, Val::Accum) },

        Expr::Ident(ident) => {
            if let Some((dtype, offset)) = locals.get(ident) {
                ((*dtype).clone(), Val::Stack(*offset))
            } else if let Some(s) = file.statics.get(ident) {
                (s.dtype.clone(), Val::Static(ident.clone(), 0))
            } else {
                panic!("Unknown identifier {}", ident);
            }
        },

        Expr::Ref(ref expr) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, frame_size, locals, output);
            ref_to_accum(frame_size, accum, &src_val, output);
            (Type::Ptr { base_type: Box::from(src_type.clone()) }, Val::Accum)
        },

        Expr::Deref(ref expr) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, frame_size, locals, output);
            match src_type {
                // Read the pointer's value into the accumulator, than change
                // the value type to to reflect this state
                Type::Ptr { base_type } => {
                    val_to_accum(frame_size, accum, &src_val, output);
                    ((*base_type).clone(), Val::AccumRef)
                },
                // In case of an array we just change the type
                Type::Array { elem_type, .. } => ((*elem_type).clone(), src_val),
                _ => panic!("Can't dereference non-reference type"),
            }
        }

        Expr::Field(ref expr, ident) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, frame_size, locals, output);
            match src_type {
                Type::Record(record) => {
                    if let Some((field_type, field_offset)) = record.fields.get(ident) {
                        (field_type.clone(),
                            field_val(accum, &src_val, *field_offset, output))
                    } else {
                        panic!("Non-existent field {} accessed", ident);
                    }
                },
                _ => panic!("Field access on non-record type"),
            }
        },

        Expr::Call(func, ref params) => {
            let (ident, rettype) = match &**func {
                Expr::Ident(ident) => {
                    if let Some(func) = file.funcs.get(ident) {
                        (ident, func.rettype.clone())
                    } else {
                        panic!("Call to undefined function {}", ident);
                    }
                },
                _ => panic!("Function name must be an identifier"),
            };
            for (i, param) in params.iter().enumerate() {
                match i {
                    0 => { let (_, val) = gen_expr(file, "rdi", param, frame_size, locals, output); val_to_accum(frame_size, "rdi", &val, output) },
                    1 => { let (_, val) = gen_expr(file, "rsi", param, frame_size, locals, output); val_to_accum(frame_size, "rsi", &val, output) },
                    2 => { let (_, val) = gen_expr(file, "rdx", param, frame_size, locals, output); val_to_accum(frame_size, "rdx", &val, output) },
                    3 => { let (_, val) = gen_expr(file, "rcx", param, frame_size, locals, output); val_to_accum(frame_size, "rcx", &val, output) },
                    4 => { let (_, val) = gen_expr(file, "r8", param, frame_size, locals, output); val_to_accum(frame_size, "r8", &val, output) },
                    5 => { let (_, val) = gen_expr(file, "r9", param, frame_size, locals, output); val_to_accum(frame_size, "r9", &val, output) },
                    _ => panic!("FIXME: too many call params"),
                };
            }
            print_or_die!(output, "xor rax, rax");
            print_or_die!(output, "call {}", ident);
            (rettype, Val::Accum)
        },

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

        /*Expr::Sub(ref expr1, ref expr2) => {
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

fn store_accum_to_val<T: std::io::Write>(accum: &str, val: &Val, output: &mut T) {
    match val {
        Val::Accum => panic!("Write to rvalue"),
        Val::AccumRef => { print_or_die!(output, "mov [r10], {}", accum); }, // NOTE: Awful hack r10 is the dest's accum
        Val::Stack(offset) => { print_or_die!(output, "mov [rsp + {}], {}", offset, accum); },
        Val::Static(ident, offset) => { print_or_die!(output, "mov [{} + {}], {}", ident, offset, accum); },
    }
}

fn gen_func<T: std::io::Write>(file: &File, func: &Func, output: &mut T) {
    print_or_die!(output, "{}:", func.name);

    // Generate locals
    let mut frame_size = 0usize;

    let mut locals = HashMap::new();
    let mut params = Vec::new();

    for (i, (name, t)) in func.params.iter().enumerate() {
        if t.get_size() > 8 {
            panic!("FIXME: larger-than register parameters!");
        }
        // Add local variable for parameter
        locals.insert(name.clone(), (t, frame_size));
        // Save index to stack offset map
        params.push((i, frame_size));
        // Increase stack frame size
        // HACK!: assume all parameters are 8 bytes for simpler moves
        frame_size += 8;
    }

    // Allocate stack frame for all auto variables too
    for stmt in &func.stmts {
        match stmt {
            Stmt::Auto(name, t, _) => {
                locals.insert(name.clone(), (t, frame_size));
                frame_size += t.get_size();
            },
            _ => (),
        }
    }

    // Align to a multiple of 16, then add 8 (sysv abi)
    frame_size = (frame_size + 15) / 16 * 16 + 8;

    // Generate function prologue
    print_or_die!(output, "push rbp\nmov rbp, rsp\nsub rsp, {}", frame_size);

    // Move parameters from registers to stack
    for param in params {
        match param {
            (0, offset) => { print_or_die!(output, "mov qword [rbp - {}], rdi", frame_size - offset); },
            (1, offset) => { print_or_die!(output, "mov qword [rbp - {}], rsi", frame_size - offset); },
            (2, offset) => { print_or_die!(output, "mov qword [rbp - {}], rdx", frame_size - offset); },
            (3, offset) => { print_or_die!(output, "mov qword [rbp - {}], rcx", frame_size - offset); },
            (4, offset) => { print_or_die!(output, "mov qword [rbp - {}], r8", frame_size - offset); },
            (5, offset) => { print_or_die!(output, "mov qword [rbp - {}], r9", frame_size - offset); },
            _ => panic!("FIXME: more than six parameters!"),
        }
    }

    for stmt in &func.stmts {
        match stmt {
            Stmt::Label(label) => {
                print_or_die!(output, ".{}:", label);
            },
            Stmt::Auto(ident, dtype, init) => {},
            Stmt::Set(ref dest, ref expr) => {
                let (ltype, lval) = gen_expr(file, "r10", dest, frame_size, &locals, output);
                let (rtype, rval) = gen_expr(file, "rax", expr, frame_size, &locals, output);
                val_to_accum(frame_size, "rax", &rval, output);
                store_accum_to_val("rax", &lval, output);
            }
            Stmt::Eval(ref expr) => {
                gen_expr(file, "rax", expr, frame_size, &locals, output);
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
    print_or_die!(output, "\nsection .bss");
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
