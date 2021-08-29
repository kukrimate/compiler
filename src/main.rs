#![feature(hash_set_entry)]

mod ast;
mod lex;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} FILE", args[0]);
        std::process::exit(1);
    }

    let data = std::fs::read_to_string(&args[1]).unwrap();
    let mut lex = lex::Lexer::new(&data);
    let mut parser = ast::Parser::new(&mut lex);

    for func in parser {
        println!("{}: {:?}", func.name, func.stmts);
    }
}
