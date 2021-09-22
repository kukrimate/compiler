// SPDX-License-Identifier: GPL-2.0-only

#![feature(hash_set_entry)]
#![feature(if_let_guard)]

mod ast;
mod gen;
mod lex;

use clap::{Arg,App};

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
        .get_matches();

    let data = std::fs::read_to_string(args.value_of("INPUT").unwrap()).unwrap();
    let file = ast::parse_file(&data);

    let path = args.value_of("output").unwrap();
    if args.occurrences_of("assembly") > 0 {
        gen::gen_asm(&file, &mut std::fs::File::create(path).unwrap());
    } else {
        gen::gen_obj(&file, &mut std::fs::File::create(path).unwrap());
    }
}
