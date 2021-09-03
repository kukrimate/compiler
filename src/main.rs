// SPDX-License-Identifier: GPL-2.0-only

#![feature(hash_set_entry)]

mod ast;
mod lex;
mod parser;

use ast::{Type,Expr,Init,Stmt,Vis,Func,File};
use std::collections::HashMap;
use std::rc::Rc;

fn gen_static_init(file: &File, init: &Init) {
    match init {
        Init::Base(expr) => match expr {
            Expr::U8(v)  => println!("db {}", v),
            Expr::I8(v)  => println!("db {}", v),
            Expr::U16(v) => println!("dw {}", v),
            Expr::I16(v) => println!("dw {}", v),
            Expr::U32(v) => println!("dd {}", v),
            Expr::I32(v) => println!("dd {}", v),
            Expr::U64(v) => println!("dq {}", v),
            Expr::I64(v) => println!("dq {}", v),
            // Static can be initialized by another static
            Expr::Ident(ident) => {
                if let Some(ref s) = file.statics.get(ident) {
                    // FIXME: fill with zeroes if the referenced static doesn't have an initializer
                    let ref refed_init = s.init.as_ref().unwrap();
                    gen_static_init(file, refed_init);
                } else {
                    panic!("Unknown")
                }
            },
            // Or by a pointer to a static (or string literal)
            Expr::Ref(dest) => {
                match &**dest {
                    Expr::Ident(ident) => println!("dq {}", ident),
                    _ => panic!("Non constant static initializer"),
                }
            },
            _ => panic!("Non constant static initializer"),
        },
        Init::List(ref list) => for i in list {
            gen_static_init(file, i);
        },
    }
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
fn ref_to_accum(accum: &str, val: &Val) {
    match val {
        Val::Accum => panic!("Rvalue used as lvalue"),
        Val::AccumRef => panic!("Rvalue used as lvalue"),
        Val::Stack(offset) => println!("lea {}, [rsp + {}]", accum, offset),
        Val::Static(ident, offset) => println!("lea {}, [{} + {}]", accum, ident, offset),
    }
}

// Load a value into the accumulator
fn val_to_accum(accum: &str, val: &Val) {
    match val {
        Val::Accum => (),
        Val::AccumRef => println!("mov {}, [{}]", accum, accum),
        Val::Stack(offset) => println!("mov {}, [rsp + {}]", accum, offset),
        Val::Static(ident, offset) => println!("mov {}, [{} + {}]", accum, ident, offset),
    }
}

fn field_val(accum: &str, val: &Val, field_offs: usize) -> Val {
    match val {
        Val::Accum => panic!("Rvalue used as lvalue"),
        Val::AccumRef => { println!("add {}, {}", accum, field_offs); Val::AccumRef },
        Val::Stack(offset) => Val::Stack(offset + field_offs),
        Val::Static(ident, offset) => Val::Static(ident.clone(), offset + field_offs),
    }
}

fn gen_expr(
        file: &File,
        accum: &str,
        in_expr: &Expr,
        locals: &HashMap<Rc<str>, (&Type, usize)>) -> (Type, Val) {

    match in_expr {
        Expr::U8(v)  => { println!("mov {}, {}", accum, v); (Type::U8, Val::Accum) },
        Expr::I8(v)  => { println!("mov {}, {}", accum, v); (Type::I8, Val::Accum) },
        Expr::U16(v) => { println!("mov {}, {}", accum, v); (Type::U16, Val::Accum) },
        Expr::I16(v) => { println!("mov {}, {}", accum, v); (Type::I16, Val::Accum) },
        Expr::U32(v) => { println!("mov {}, {}", accum, v); (Type::U32, Val::Accum) },
        Expr::I32(v) => { println!("mov {}, {}", accum, v); (Type::I32, Val::Accum) },
        Expr::U64(v) => { println!("mov {}, {}", accum, v); (Type::U64, Val::Accum) },
        Expr::I64(v) => { println!("mov {}, {}", accum, v); (Type::I64, Val::Accum) },

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
            let (src_type, src_val) = gen_expr(file, accum, expr, locals);
            ref_to_accum(accum, &src_val);
            (Type::Ptr { base_type: Box::from(src_type.clone()) }, Val::Accum)
        },

        Expr::Deref(ref expr) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, locals);
            match src_type {
                // Read the pointer's value into the accumulator, than change
                // the value type to to reflect this state
                Type::Ptr { base_type } => {
                    val_to_accum(accum, &src_val);
                    ((*base_type).clone(), Val::AccumRef)
                },
                // In case of an array we just change the type
                Type::Array { elem_type, .. } => ((*elem_type).clone(), src_val),
                _ => panic!("Can't dereference non-reference type"),
            }
        }

        Expr::Field(ref expr, ident) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, locals);
            match src_type {
                Type::Record(record) => {
                    if let Some((field_type, field_offset)) = record.fields.get(ident) {
                        (field_type.clone(),
                            field_val(accum, &src_val, *field_offset))
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
                    0 => { let (_, val) = gen_expr(file, "rdi", param, locals); val_to_accum("rdi", &val) },
                    1 => { let (_, val) = gen_expr(file, "rsi", param, locals); val_to_accum("rsi", &val) },
                    2 => { let (_, val) = gen_expr(file, "rdx", param, locals); val_to_accum("rdx", &val) },
                    3 => { let (_, val) = gen_expr(file, "rcx", param, locals); val_to_accum("rcx", &val) },
                    4 => { let (_, val) = gen_expr(file, "r8", param, locals); val_to_accum("r8", &val) },
                    5 => { let (_, val) = gen_expr(file, "r9", param, locals); val_to_accum("r9", &val) },
                    _ => panic!("FIXME: too many call params"),
                };
            }
            println!("call {}", ident);
            (rettype, Val::Accum)
        },

        Expr::Inv(ref expr) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, locals);
            assert_integral(&src_type);
            val_to_accum(accum, &src_val);
            println!("not {}", accum);
            (src_type, Val::Accum)
        },
        Expr::Neg(ref expr) => {
            let (src_type, src_val) = gen_expr(file, accum, expr, locals);
            assert_integral(&src_type);
            val_to_accum(accum, &src_val);
            println!("neg {}", accum);
            (src_type, Val::Accum)
        },

        /*Expr::Add(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
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

fn gen_func(file: &File, func: &Func) {
    println!("{}:", func.name);
    println!("{:#?}", func.stmts);

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
    println!("push rbp\nmov rbp, rsp\nsub rsp, {}", frame_size);

    // Move parameters from registers to stack
    for param in params {
        match param {
            (0, offset) => println!("mov qword [rsp + {}], rdi", offset),
            (1, offset) => println!("mov qword [rsp + {}], rsi", offset),
            (2, offset) => println!("mov qword [rsp + {}], rdx", offset),
            (3, offset) => println!("mov qword [rsp + {}], rcx", offset),
            (4, offset) => println!("mov qword [rsp + {}], r8", offset),
            (5, offset) => println!("mov qword [rsp + {}], r9", offset),
            _ => panic!("FIXME: more than six parameters!"),
        }
    }

    for stmt in &func.stmts {
        match stmt {
            Stmt::Eval(ref expr) => {
                gen_expr(file, "rax", expr, &locals);
            },
            Stmt::Ret(_) => println!("jmp done"),
            _ => todo!("statement {:?}", stmt),
        }
    }

    // Generate function epilogue
    println!("done:\nleave\nret");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} FILE", args[0]);
        std::process::exit(1);
    }

    let data = std::fs::read_to_string(&args[1]).unwrap();
    let file = parser::parse_file(&data);

    let mut exports = Vec::new();
    let mut externs = Vec::new();

    let mut bss = HashMap::new();

    // Generate data
    println!("section .data");
    for (_, cur_static) in &file.statics {
        match cur_static.vis {
            Vis::Export => exports.push(cur_static.name.clone()),
            Vis::Extern => externs.push(cur_static.name.clone()),
            _ => (),
        };

        match &cur_static.init {
            // Generate static initializer in .data
            Some(ref init) => {
                println!("{}:", cur_static.name);
                gen_static_init(&file, init);
            },
            // Take note for bss allocation later
            None => {
                bss.insert(cur_static.name.clone(),
                    cur_static.dtype.get_size());
            },
        };
    }

    // Generate bss
    println!("\nsection .bss");
    for (name, len) in bss {
        println!("{}: resb {}", name, len);
    }

    // Generate functions
    println!("section .text");

    for (_, func) in &file.funcs {
        match func.vis {
            Vis::Export => exports.push(func.name.clone()),
            Vis::Extern => externs.push(func.name.clone()),
            _ => (),
        };
        // Generate body for non-extern function
        if func.vis != Vis::Extern {
            gen_func(&file, &func);
        }
    }

    // Generate markers
    for exp in exports {
        println!("global {}", exp);
    }
    for ext in externs {
        println!("extern {}", ext);
    }
}
