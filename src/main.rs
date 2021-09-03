// SPDX-License-Identifier: GPL-2.0-only

#![feature(hash_set_entry)]

mod ast;
mod lex;
mod parser;

use ast::{Type,Expr,Init,Stmt,Vis,Func,File};
use std::collections::HashMap;
use std::rc::Rc;

fn gen_static_init(init: &Init) {
    match init {
        Init::Base(ref expr) => match expr {
            Expr::U8(v)  => println!("db {}", v),
            Expr::I8(v)  => println!("db {}", v),
            Expr::U16(v) => println!("dw {}", v),
            Expr::I16(v) => println!("dw {}", v),
            Expr::U32(v) => println!("dd {}", v),
            Expr::I32(v) => println!("dd {}", v),
            Expr::U64(v) => println!("dq {}", v),
            Expr::I64(v) => println!("dq {}", v),
            _ => panic!("FIXME: Static initializer must be a constant"),
        },
        Init::List(ref list) => for i in list {
            gen_static_init(i);
        },
    }
}

//
// Target of a data reference
//
/*
enum RefTgt {
    Immed(Expr),              // Immediate value
    Static(Rc<str>),            // Global variable
    Stack(usize),               // Local variable
    Offset(Box<RefTgt>, usize), // Offset from another reference
}
*/

fn gen_expr(
        file: &File,
        accum: &str,
        in_expr: &Expr,
        locals: &HashMap<Rc<str>, (&Type, usize)>) {
    match in_expr {
        Expr::U8(v)  => { println!("mov {}, {}", accum, v) },
        Expr::I8(v)  => { println!("mov {}, {}", accum, v) },
        Expr::U16(v) => { println!("mov {}, {}", accum, v) },
        Expr::I16(v) => { println!("mov {}, {}", accum, v) },
        Expr::U32(v) => { println!("mov {}, {}", accum, v) },
        Expr::I32(v) => { println!("mov {}, {}", accum, v) },
        Expr::U64(v) => { println!("mov {}, {}", accum, v) },
        Expr::I64(v) => { println!("mov {}, {}", accum, v) },

        Expr::Ident(ident) => {
            if let Some((t, offset)) = locals.get(ident) {
                todo!("local variable reference");
            } else {
                // FIXME: don't assume it's always a global and take its address
                println!("mov {}, {}", accum, ident)
            }
        },

        /*Expr::Ref(ref expr) => gen_expr(expr, locals, data),
        Expr::Deref(ref expr) => gen_expr(expr, locals, data),
        Expr::Field(ref expr, _) => gen_expr(expr, locals, data),
        Expr::Elem(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },*/
        Expr::Call(func, ref params) => {
            for (i, param) in params.iter().enumerate() {
                match i {
                    0 => gen_expr(file, "rdi", param, locals),
                    1 => gen_expr(file, "rsi", param, locals),
                    2 => gen_expr(file, "rdx", param, locals),
                    3 => gen_expr(file, "rcx", param, locals),
                    4 => gen_expr(file, "r8", param, locals),
                    5 => gen_expr(file, "r9", param, locals),
                    _ => panic!("FIXME: too many call params"),
                }
            }
            match &**func {
                Expr::Ident(name) => println!("call {}", name),
                _ => panic!("Function name must be an identifier"),
            }
        },

        /*
        Expr::Inv(ref expr) => gen_expr(expr, locals, data),
        Expr::Neg(ref expr) => gen_expr(expr, locals, data),
        Expr::Add(ref expr1, ref expr2) => {
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
        Expr::Cast(ref expr, _) => gen_expr(expr, locals, data),
        */
        _ => todo!("expression {:?}", in_expr),
    }
}

fn gen_func(file: &File, func: &Func) {
    println!("{}:", func.name);

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
                gen_expr(file, "rax", expr, &locals)
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
    for cur_static in &file.statics {
        match cur_static.vis {
            Vis::Export => exports.push(cur_static.name.clone()),
            Vis::Extern => externs.push(cur_static.name.clone()),
            _ => (),
        };

        match &cur_static.init {
            // Generate static initializer in .data
            Some(ref init) => {
                println!("{}:", cur_static.name);
                gen_static_init(init);
            },
            // Take note for bss allocation later
            None => {
                bss.insert(cur_static.name.clone(),
                    cur_static.r#type.get_size());
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

    for func in &file.funcs {
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
    println!("");
    for exp in exports {
        println!("global {}", exp);
    }
    for ext in externs {
        println!("extern {}", ext);
    }
}
