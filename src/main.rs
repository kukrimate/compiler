// SPDX-License-Identifier: GPL-2.0-only

#![feature(hash_set_entry)]

mod ast;
mod lex;

use ast::{IntVal,Vis,Init,Func,Stmt,Expr};
use std::collections::HashMap;
use std::rc::Rc;

fn gen_intval(intval: &IntVal, dest: &mut Vec<u8>) {
    match intval {
        IntVal::U8(v)  => dest.extend(&v.to_le_bytes()),
        IntVal::I8(v)  => dest.extend(&v.to_le_bytes()),
        IntVal::U16(v) => dest.extend(&v.to_le_bytes()),
        IntVal::I16(v) => dest.extend(&v.to_le_bytes()),
        IntVal::U32(v) => dest.extend(&v.to_le_bytes()),
        IntVal::I32(v) => dest.extend(&v.to_le_bytes()),
        IntVal::U64(v) => dest.extend(&v.to_le_bytes()),
        IntVal::I64(v) => dest.extend(&v.to_le_bytes()),
    }
}

fn gen_static_init(init: &Init, dest: &mut Vec<u8>) {
    match init {
        Init::Base(ref expr) => match expr {
            Expr::Const(intval) => gen_intval(intval, dest),
            _ => panic!("FIXME: Static initializer must be a constant"),
        },
        Init::List(ref list) => for i in list {
            gen_static_init(i, dest);
        },
    }
}

// enum Loc {
//     Static(Rc<str>),    // Global variable
//     Stack(usize),       // Local variable
//     Offset,
// }

static mut LITNO: usize = 0;

fn gen_expr(
        accum: &str,
        in_expr: &Expr,
        locals: &HashMap<Rc<str>, usize>,
        data: &mut HashMap<Rc<str>, Vec<u8>>) {
    match in_expr {
        Expr::Const(intval) => println!("mov {}, {}", accum, intval.as_usize()),
        // Expr::Ident(_) => (),
        Expr::Str(_, s) => {
            let name: Rc<str>;
            unsafe {
                name = Rc::from(format!("slit_{}", LITNO));
                LITNO += 1;
            };
            let mut utf8data = Vec::new();
            utf8data.extend(s.as_bytes());
            utf8data.push(0);
            data.insert(name.clone(), utf8data);
            println!("mov {}, {}", accum, name);
        },
        /*Expr::Ref(ref expr) => gen_expr(expr, locals, data),
        Expr::Deref(ref expr) => gen_expr(expr, locals, data),
        Expr::Inv(ref expr) => gen_expr(expr, locals, data),
        Expr::Neg(ref expr) => gen_expr(expr, locals, data),
        Expr::Field(ref expr, _) => gen_expr(expr, locals, data),
        Expr::Elem(ref expr1, ref expr2) => {
            gen_expr(expr1, locals, data);
            gen_expr(expr2, locals, data);
        },*/
        Expr::Call(func, ref params) => {
            for (i, param) in params.iter().enumerate() {
                match i {
                    0 => gen_expr("rdi", param, locals, data),
                    1 => gen_expr("rsi", param, locals, data),
                    2 => gen_expr("rdx", param, locals, data),
                    3 => gen_expr("rcx", param, locals, data),
                    4 => gen_expr("r8", param, locals, data),
                    5 => gen_expr("r9", param, locals, data),
                    _ => panic!("FIXME: too many call params"),
                }
            }
            match &**func {
                Expr::Ident(name) => println!("call {}", name),
                _ => panic!("Function name must be an identifier"),
            }
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
        Expr::Cast(ref expr, _) => gen_expr(expr, locals, data),
        */
        _ => todo!("expression {:?}", in_expr),
    }
}

fn gen_func(func: &Func, data: &mut HashMap<Rc<str>, Vec<u8>>) {
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
        locals.insert(name.clone(), frame_size);
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
                locals.insert(name.clone(), frame_size);
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
                gen_expr("rax", expr, &locals, data)
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

    let mut lex = lex::Lexer::new(&data);
    let mut parser = ast::Parser::new(&mut lex);
    let (statics, funcs) = parser.parse_file();

    let mut exports = Vec::new();
    let mut externs = Vec::new();

    let mut data = HashMap::new();
    let mut bss = HashMap::new();

    // Collect statics
    for cur_static in statics {
        match cur_static.vis {
            Vis::Export => exports.push(cur_static.name.clone()),
            Vis::Extern => externs.push(cur_static.name.clone()),
            _ => (),
        };

        match cur_static.init {
            // Generate static initializer in .data
            Some(init) => {
                let mut dest = Vec::new();
                gen_static_init(&init, &mut dest);
                data.insert(cur_static.name.clone(), dest);
            },
            // Allocate space in .bss
            None => {
                bss.insert(cur_static.name.clone(),
                    cur_static.r#type.get_size());
            },
        };
    }

    // Collect and tame functions
    println!("section .text");

    for cur_func in funcs {
        match cur_func.vis {
            Vis::Export => exports.push(cur_func.name.clone()),
            Vis::Extern => externs.push(cur_func.name.clone()),
            _ => (),
        };
        // Generate body for non-extern function
        if cur_func.vis != Vis::Extern {
            gen_func(&cur_func, &mut data);
        }
    }

    // Generate data
    println!("\nsection .data");
    for (name, bytes) in data {
        if bytes.len() > 0 {
            print!("{}: db ", name);
        } else {
            print!("{}:", name);
        }

        for b in bytes {
            print!("{}, ", b);
        }
        println!("");
    }

    // Generate bss
    println!("\nsection .bss");
    for (name, len) in bss {
        println!("{}: resb {}", name, len);
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
