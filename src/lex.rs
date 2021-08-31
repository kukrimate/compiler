use logos;
use std::collections::HashSet;
use std::rc::Rc;

static mut SYMTAB: Option<HashSet<Rc<str>>> = None;

// Exposed lexer type
pub type Lexer<'source> = logos::Lexer<'source, Token>;

fn bin(lex: &mut Lexer) -> usize {
    let mut result = 0usize;
    for ch in &lex.slice().as_bytes()[2..] {
        result = (result << 1) + (ch - b'0') as usize;
    }
    result
}

fn oct(lex: &mut Lexer) -> usize {
    let mut result = 0usize;
    for ch in &lex.slice().as_bytes()[2..] {
        result = (result << 3) + (ch - b'0') as usize;
    }
    result
}

fn dec(lex: &mut Lexer) -> usize {
    let mut result = 0usize;
    for ch in lex.slice().as_bytes() {
        result = result * 10 + (ch - b'0') as usize;
    }
    result
}

fn hex(lex: &mut Lexer) -> usize {
    let mut result = 0usize;
    for ch in &lex.slice().as_bytes()[2..] {
        match ch {
            b'0'..=b'9' => {
                result = (result << 4) + (ch - b'0') as usize;
            },
            b'a'..=b'f' => {
                result = (result << 4) + (ch - b'a' + 0xa) as usize;
            },
            b'A'..=b'F' => {
                result = (result << 4) + (ch - b'A' + 0xa) as usize;
            }
            _ => panic!(),
        }
    }
    result
}

fn ch(lex: &mut Lexer) -> usize {
    // FIXME: parse escape sequences
    let mut result = 0usize;
    for ch in lex.slice().as_bytes() {
        result = result << 8 | *ch as usize;
    }
    result
}

fn symtab_put(lex: &mut Lexer, s: &str) -> Rc<str> {
    unsafe {
        if let None = SYMTAB {
            SYMTAB = Some(HashSet::new());
        }
        SYMTAB.as_mut().unwrap().get_or_insert(s.into()).clone()
    }
}

fn str(lex: &mut Lexer) -> Rc<str> {
    // FIXME: parse escape sequences
    let s = lex.slice();
    symtab_put(lex, &s[1..s.len() - 1])
}

fn ident(lex: &mut Lexer) -> Rc<str> {
    symtab_put(lex, &lex.slice()[1..])
}

#[derive(logos::Logos, Debug)]
pub enum Token {
    // Typenames
    #[token("u8")]
    U8,
    #[token("i8")]
    I8,
    #[token("u16")]
    U16,
    #[token("i16")]
    I16,
    #[token("u32")]
    U32,
    #[token("i32")]
    I32,
    #[token("u64")]
    U64,
    #[token("i64")]
    I64,

    // Declarations
    #[token("record")]
    Record,
    #[token("union")]
    Union,
    #[token("fn")]
    Fn,
    #[token("auto")]
    Auto,
    #[token("static")]
    Static,

    // Declaration markers
    #[token("export")]
    Export,

    // Constants
    #[regex(r"0b[0-1]+", bin)]
    #[regex(r"0o[0-7]+", oct)]
    #[regex(r"[0-9]+", dec)]
    #[regex(r"0x[0-9a-fA-F]+", hex)]
    #[regex(r"'(\\.|[^\\'])*'", ch)]
    Constant(usize),

    #[regex(r#""(\\.|[^\\"])*""#, str)]
    Str(Rc<str>),

    // Indentifiers, labels
    #[regex(r"\$[a-zA-Z_][a-zA-Z0-9_]*", ident)]
    Ident(Rc<str>),
    #[regex(r"@[a-zA-Z_][a-zA-Z0-9_]*", ident)]
    Label(Rc<str>),

    // Symbols
    #[token("=")]
    Eq,        // =
    #[token("~")]
    Tilde,     // ~
    #[token("+")]
    Add,       // +
    #[token("-")]
    Sub,       // -
    #[token("*")]
    Mul,       // *
    #[token("/")]
    Div,       // /
    #[token("%")]
    Rem,       // %
    #[token("|")]
    Or,        // |
    #[token("&")]
    And,       // &
    #[token("^")]
    Xor,       // ^
    #[token("<<")]
    Lsh,       // <<
    #[token(">>")]
    Rsh,       // >>
    #[token("(")]
    LParen,    // (
    #[token(")")]
    RParen,    // )
    #[token("[")]
    LSq,       // [
    #[token("]")]
    RSq,       // ]
    #[token("{")]
    LCurly,    // {
    #[token("}")]
    RCurly,    // }
    #[token(";")]
    Semicolon, // ;
    #[token(":")]
    Colon,     // :
    #[token(",")]
    Comma,     // ,
    #[token("as")]
    Cast,      // cast operator

    // Statements
    #[token("set")]
    Set,
    #[token("jmp")]
    Jmp,
    #[token("jeq")]
    Jeq,
    #[token("jl")]
    Jl,
    #[token("jle")]
    Jle,
    #[token("jg")]
    Jg,
    #[token("jge")]
    Jge,

    #[error]
    #[regex(r"[ \t\n\f\v]+", logos::skip)]
    #[regex(r"//.*\n", logos::skip)]
    Error,
}
