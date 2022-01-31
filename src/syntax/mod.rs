mod ast;
mod parser;
mod symtab;

use crate::lex::Lexer;
use crate::gen::Gen;

pub use ast::{Ty,ExprKind,Expr,Stmt};
pub use symtab::{Vis,SymTab};


pub fn parse_file(data: &str, gen: &mut Gen) {
    let mut parser = parser::Parser::new(Lexer::new(data), gen);
    parser.process();
}
