// SPDX-License-Identifier: GPL-2.0-only

mod ast;
mod gen;
mod lex;
mod lower;

use clap::{Arg,App};
use std::process::Command;
use tempfile::NamedTempFile;

fn assemble(asm_path: &str, obj_path: &str) {
    let status = Command::new("nasm")
        .args(["-f", "elf64", "-o", obj_path, asm_path])
        .status()
        .expect("failed to run nasm");

    if !status.success() {
        panic!("assembly with nasm failed");
    }
}

fn link(obj_path: &str, exe_path: &str) {
    // NOTE: C compiler is obviously only used for linking
    // we are not compiling C here, but we do want to link against libc
    // and the C startup files
    let status = Command::new("cc")
        .args(["-no-pie", "-o", exe_path, obj_path])
        .status()
        .expect("failed to run cc");

    if !status.success() {
        panic!("linking failed");
    }
}

fn main() {
    let args = App::new("compiler")
        .version("1.0.0")
        .author("Mate Kukri <km@mkukri.xyz>")
        .about("Compiler for a terrible programming language")
        .arg(Arg::with_name("INPUT")
            .help("Input file")
            .required(true)
            .index(1))
        .arg(Arg::with_name("output")
            .short("o")
            .long("output")
            .value_name("OUTPUT")
            .help("Output file")
            .required(true)
            .takes_value(true))
        .arg(Arg::with_name("assembly")
            .short("a")
            .long("assembly")
            .help("Generate assembly instead of an object file"))
        .arg(Arg::with_name("compile")
            .short("c")
            .long("compile")
            .help("Generate an object file instead of an executable"))
        .get_matches();

    let data = std::fs::read_to_string(args.value_of("INPUT").unwrap()).unwrap();
    let output_path = args.value_of("output").unwrap();

    let mut gen = gen::nasm::GenNasm::new();
    ast::parse_file(&data, &mut gen);

    if args.occurrences_of("assembly") > 0 {
        // Just write assembly to output
        gen.finalize(&mut std::fs::File::create(output_path).unwrap());
    } else {
        // Write assembly to tempfile
        let mut asm_file = NamedTempFile::new().unwrap();
        gen.finalize(&mut asm_file);
        if args.occurrences_of("compile") > 0 {
            // Assemble to output directly
            assemble(asm_file.path().to_str().unwrap(), output_path);
        } else {
            // Assembly to tempfile than link
            let obj_file = NamedTempFile::new().unwrap();
            assemble(asm_file.path().to_str().unwrap(),
                        obj_file.path().to_str().unwrap());
            link(obj_file.path().to_str().unwrap(), output_path);
        }
    }
}
