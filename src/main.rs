// SPDX-License-Identifier: GPL-2.0-only

mod ast;
mod gen;
mod lex;
mod lower;

use clap::{Arg,App};

fn main() {
    let args = App::new("compiler")
        .version("1.0.0")
        .author("Mate Kukri <km@mkukri.xyz>")
        .about("K compiler for Sigma16")
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
        .get_matches();

    let data = std::fs::read_to_string(args.value_of("INPUT").unwrap()).unwrap();
    let output_path = args.value_of("output").unwrap();

    let mut gen = gen::s16::GenS16::new();
    ast::parse_file(&data, &mut gen);

    // Just write assembly to output
    gen.finalize(&mut std::fs::File::create(output_path).unwrap());
}
