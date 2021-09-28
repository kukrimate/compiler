// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation
//

use super::ast;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::process::Command;
use std::rc::Rc;
use tempfile::tempdir;

// Code generation context
struct Gen<'a> {
    file: &'a ast::File,
    output: &'a mut fs::File,
    exports: Vec<Rc<str>>,
    externs: Vec<Rc<str>>,
}

impl<'a> Gen<'a> {
    fn new(file: &'a ast::File, output: &'a mut fs::File) -> Gen<'a> {
        Gen {
            file: file,
            output: output,
            exports: Vec::new(),
            externs: Vec::new(),
        }
    }

    // Generate assembly code for a static initializer
    fn gen_static_init(&mut self, dtype: &Rc<ast::Type>, init: &ast::Init) {
        match &**dtype {
            ast::Type::Array { elem_type, elem_count } => {
                let init_list = init.want_list();
                if init_list.len() != *elem_count {
                    panic!("Invalid static array initializer");
                }
                for elem_init in init_list {
                    self.gen_static_init(elem_type, elem_init);
                }
            },

            ast::Type::Record { is_union, fields, size, .. } => {
                if *is_union {
                    todo!("union initializer");
                } else {
                    let mut offset = 0;
                    for ((field_type, field_offset), field_init)
                                        in fields.iter().zip(init.want_list()) {
                        // There might be padding before each record field
                        if offset < *field_offset {
                            writeln!(self.output, "times {} db 0xCC", *field_offset - offset).unwrap();
                        }
                        self.gen_static_init(field_type, field_init);
                        offset = *field_offset + field_type.get_size();
                    }
                    // Record might have padding at the end
                    if offset < *size {
                        writeln!(self.output, "times {} db 0xCC", *size - offset).unwrap();
                    }
                }
            },

            _ => {
                match init.want_expr() {
                    // Static can be initialized by a constant
                    ast::Expr::Const(ctype, val) => {
                        if dtype != ctype {
                            panic!("Type mismatch for static initializer")
                        }

                        match &**dtype {
                            ast::Type::U8 | ast::Type::I8
                                => writeln!(self.output, "db 0x{:X}", *val as u8).unwrap(),
                            ast::Type::U16 | ast::Type::I16
                                => writeln!(self.output, "dw 0x{:X}", *val as u16).unwrap(),
                            ast::Type::U32 | ast::Type::I32
                                => writeln!(self.output, "dd 0x{:X}", *val as u32).unwrap(),
                            ast::Type::U64 | ast::Type::I64 | ast::Type::Ptr {..}
                                => writeln!(self.output, "dq 0x{:X}", *val as u64).unwrap(),

                            _ => panic!("Invalid type {:?} for constant", dtype)
                        }
                    }

                    // Or by the address of another static
                    ast::Expr::Ref(expr) if let ast::Expr::Ident(ident) = &**expr
                        => match &**dtype {
                            // FIXME: check base type of pointer too
                            ast::Type::Ptr {..} => writeln!(self.output, "dq {}", ident).unwrap(),
                            _ => panic!("Non-pointer static initialized with pointer"),
                        },

                    _ => panic!("Non constant static initializer")
                }
            }
        }
    }

    // Generate all static initializers
    fn gen_statics(&mut self) {
        let mut bss = HashMap::new();

        // Generate data
        {
            let mut offset = 0;
            writeln!(self.output, "section .data").unwrap();
            for (name, def) in &self.file.statics {
                match def.vis {
                    ast::Vis::Export => self.exports.push(name.clone()),
                    ast::Vis::Extern => self.externs.push(name.clone()),
                    _ => (),
                }

                if let Some(init) = &def.init {
                    // Generate padding for alignment
                    let begin = round_up!(offset, def.dtype.get_align());
                    if offset < begin {
                        writeln!(self.output, "times {} db 0xcc", begin - offset).unwrap();
                    }
                    // Generate static initializer in .data
                    writeln!(self.output, "{}:", name).unwrap();
                    self.gen_static_init(&def.dtype, init);
                    offset = begin + def.dtype.get_size();
                } else {
                    // Take note for bss allocation later
                    bss.insert(name.clone(), &def.dtype);
                }
            }
        }

        // Generate bss
        {
            let mut offset = 0;
            writeln!(self.output, "section .bss").unwrap();
            for (name, dtype) in bss {
                // Generate padding for alignment
                let begin = round_up!(offset, dtype.get_align());
                if offset < begin {
                    writeln!(self.output, "resb {}", begin - offset).unwrap();
                }
                // Reserve space for un-initialzed object
                writeln!(self.output, "{}: resb {}", name, dtype.get_size()).unwrap();
                offset += begin + dtype.get_size();
            }
        }
    }
}

/*
fn gen_func<T: std::io::Write>(file: &File, func: &Func, output: &mut T) {
    print_or_die!(output, "{}:", func.name);

    // Create context
    let mut ctx = FuncCtx {
        usedregs: 0,
        asmcode: String::new(),
    };

    // Allocate stack slots for parameters
    for (name, dtype) in &func.params {
        if dtype.get_size() > 8 {
            todo!("larger-than register parameters");
        }
        // Add local variable for parameter
        ctx.locals.insert(name.clone(), (dtype.clone(), ctx.frame_size));
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

    // Align stack frame size to a multiple of 16 + 8 (SysV ABI requirement)
    ctx.frame_size = (ctx.frame_size + 15) / 16 * 16 + 8;
    // Generate function prologue
    print_or_die!(output, "push rbp\nmov rbp, rsp\nsub rsp, {}", ctx.frame_size);

    // Generate function epilogue
    print_or_die!(output, ".$done:\nleave\nret");
}

*/

pub fn gen_asm(file: &ast::File, output: &mut fs::File) {
    let mut gen = Gen::new(file, output);

    gen.gen_statics();
}

pub fn gen_obj(input: &ast::File, output: &mut fs::File) {
    let tdir = tempdir().unwrap();
    let tasm = tdir.path().join("asm");
    let tobj = tdir.path().join("obj");

    // Generate assembly
    gen_asm(input, &mut fs::File::create(&tasm).unwrap());

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
