// SPDX-License-Identifier: GPL-2.0-only

//
// Lexical analyzer using logos
//

use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone,Debug)]
pub enum Token {
    // Typenames
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    USize,

    // Boolean constants
    True,
    False,

    // Declarations
    Record,
    Fn,
    Static,
    Let,

    // Declaration markers
    Export,
    Extern,

    // Constants
    Constant(usize),
    Str(Rc<str>),

    // Indentifiers
    Ident(Rc<str>),

    // Symbols
    Assign,    // =
    Tilde,     // ~
    Excl,      // !
    Mul,       // *
    Div,       // /
    Rem,       // %
    Add,       // +
    Sub,       // -
    Lsh,       // <<
    Rsh,       // >
    Lt,        // <
    Le,        // <=
    Gt,        // >
    Ge,        // >=
    Eq,        // ==
    Ne,        // !=
    And,       // &
    Xor,       // ^
    Or,        // |
    LAnd,      // &&
    LOr,       // ||
    LParen,    // (
    RParen,    // )
    LSq,       // [
    RSq,       // ]
    LCurly,    // {
    RCurly,    // }
    Semicolon, // ;
    Colon,     // :
    Comma,     // ,
    Dot,       // .
    As,        // cast operator
    Arrow,     // function return type
    Varargs,   // variable arguments marker

    // Statements
    Ret,
    Jmp,
    If,
    Else,
    While,

    Error(u8),
}

pub struct Lexer<'a> {
    kws: HashMap<&'a str, Token>,
    data: &'a [u8],
}

impl<'a> Lexer<'a> {
    pub fn new(data: &'a str) -> Lexer<'a> {
        let mut kws = HashMap::new();
        kws.insert("bool",      Token::Bool);
        kws.insert("u8",        Token::U8);
        kws.insert("i8",        Token::I8);
        kws.insert("u16",       Token::U16);
        kws.insert("i16",       Token::I16);
        kws.insert("u32",       Token::U32);
        kws.insert("i32",       Token::I32);
        kws.insert("u64",       Token::U64);
        kws.insert("i64",       Token::I64);
        kws.insert("usize",     Token::USize);
        kws.insert("true",      Token::True);
        kws.insert("false",     Token::False);
        kws.insert("record",    Token::Record);
        kws.insert("fn",        Token::Fn);
        kws.insert("static",    Token::Static);
        kws.insert("let",       Token::Let);
        kws.insert("export",    Token::Export);
        kws.insert("extern",    Token::Extern);
        kws.insert("as",        Token::As);
        kws.insert("ret",       Token::Ret);
        kws.insert("jmp",       Token::Jmp);
        kws.insert("if",        Token::If);
        kws.insert("else",      Token::Else);
        kws.insert("while",     Token::While);
        Lexer {
            kws: kws,
            data: data.as_bytes()
        }
    }

    fn look(&mut self, i: usize) -> Option<u8> {
        self.data.get(i).cloned()
    }

    fn eat(&mut self, n: usize) {
        self.data = &self.data[n..]
    }

    fn unescape(&mut self) -> char {
        match self.look(0).expect("Missing escape sequence") {
            b'n' => { self.eat(1); '\n' }
            b'r' => { self.eat(1); '\r' }
            b't' => { self.eat(1); '\t' }
            b'0' => { self.eat(1); '\0' }
            b'x' | b'u' => {
                let mut val = 0;
                self.eat(1);
                loop {
                    match self.look(0) {
                        Some(byte @ (b'0'..=b'9')) => val = val << 4 | (byte - b'0') as u32,
                        Some(byte @ (b'a'..=b'f')) => val = val << 4 | (byte - b'a' + 0xa) as u32,
                        Some(byte @ (b'A'..=b'F')) => val = val << 4 | (byte - b'A' + 0xa) as u32,
                        _ => break,
                    }
                    self.eat(1);
                }
                std::char::from_u32(val).expect("Escape sequence must be valid unicode")
            },
            _ => panic!("Unknown escape sequence"),
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        Some(loop {
            break match self.look(0)? {
                // Whitespace
                b' ' | b'\n' | b'\r' | b'\t' => { self.eat(1); continue },
                // Identifier
                b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
                    let begin = self.data;

                    // First character already matched
                    self.eat(1);

                    // Consume characters until a non-matching one is hit
                    while let Some(b'_' | b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9') = self.look(0) {
                        self.eat(1);
                    }

                    let slice = unsafe { std::str::from_utf8_unchecked(&begin[.. begin.len() - self.data.len()]) };
                    if let Some(tok) = self.kws.get(slice) {
                        tok.clone()
                    } else {
                        Token::Ident(slice.into())
                    }
                },
                // Numbers
                b'0'..=b'9' => {
                    let mut val = 0;
                    match self.look(1) {
                        Some(b'b') => {
                            self.eat(2);
                            loop {
                                match self.look(0) {
                                    Some(byte @ (b'0'..=b'1')) => val = val << 1 | (byte - b'0') as usize,
                                    _ => break,
                                }
                                self.eat(1);
                            }
                        },
                        Some(b'o') => {
                            self.eat(2);
                            loop {
                                match self.look(0) {
                                    Some(byte @ (b'0'..=b'7')) => val = val << 3 | (byte - b'0') as usize,
                                    _ => break,
                                }
                                self.eat(1);
                            }
                        },
                        Some(b'x') => {
                            self.eat(2);
                            loop {
                                match self.look(0) {
                                    Some(byte @ (b'0'..=b'9')) => val = val << 4 | (byte - b'0') as usize,
                                    Some(byte @ (b'a'..=b'f')) => val = val << 4 | (byte - b'a' + 0xa) as usize,
                                    Some(byte @ (b'A'..=b'F')) => val = val << 4 | (byte - b'A' + 0xa) as usize,
                                    _ => break,
                                }
                                self.eat(1);
                            }
                        },
                        _ => {
                            loop {
                                match self.look(0) {
                                    Some(byte @ (b'0'..=b'9')) => val = val * 10 + (byte - b'0') as usize,
                                    _ => break,
                                }
                                self.eat(1);
                            }
                        }
                    }
                    Token::Constant(val)
                },
                // Character constant
                b'\'' => {
                    let mut v = Vec::new();
                    self.eat(1);
                    loop {
                        match self.look(0) {
                            Some(b'\\') => {
                                self.eat(1);
                                // NOTE: the part before the escape must be valid
                                // UTF-8, thus we can do this hackery
                                let mut s = unsafe { String::from_utf8_unchecked(v) };
                                s.push(self.unescape());
                                v = s.into_bytes();
                            },
                            Some(b'\'')  => {
                                self.eat(1);
                                break;
                            },
                            Some(byte)  => {
                                self.eat(1);
                                v.push(byte);
                            },
                            None        => panic!("Unterminated char constant"),
                        }
                    }
                    let s = unsafe { String::from_utf8_unchecked(v) };
                    let mut chars = s.chars();
                    let t = Token::Constant(chars.next().expect("Empty char constant") as usize);
                    if let Some(_) = chars.next() {
                        panic!("Char constant must contain only one codepoint")
                    }
                    t
                },
                // String literal
                b'"' => {
                    let mut v = Vec::new();
                    self.eat(1);
                    loop {
                        match self.look(0) {
                            Some(b'\\') => {
                                self.eat(1);
                                // NOTE: the part before the escape must be valid
                                // UTF-8, thus we can do this hackery
                                let mut s = unsafe { String::from_utf8_unchecked(v) };
                                s.push(self.unescape());
                                v = s.into_bytes();
                            },
                            Some(b'"')  => {
                                self.eat(1);
                                break;
                            },
                            Some(byte)  => {
                                self.eat(1);
                                v.push(byte);
                            },
                            None        => panic!("Unterminated string literal"),
                        }
                    }
                    Token::Str(unsafe { String::from_utf8_unchecked(v) }.into())
                },
                // Symbols
                b'.' => match (self.look(1), self.look(2)) {
                    (Some(b'.'), Some(b'.')) => { self.eat(3); Token::Varargs },
                    _                        => { self.eat(1); Token::Dot },
                },
                b'=' => match self.look(1) {
                    Some(b'=') => { self.eat(2); Token::Eq },
                    _          => { self.eat(1); Token::Assign },
                },
                b'!' => match self.look(1) {
                    Some(b'=') => { self.eat(2); Token::Ne },
                    _          => { self.eat(1); Token::Excl },
                },
                b'<' => match self.look(1) {
                    Some(b'<') => { self.eat(2); Token::Lsh },
                    Some(b'=') => { self.eat(2); Token::Le },
                    _          => { self.eat(1); Token::Lt },
                },
                b'>' => match self.look(1) {
                    Some(b'>') => { self.eat(2); Token::Rsh },
                    Some(b'=') => { self.eat(2); Token::Ge },
                    _          => { self.eat(1); Token::Gt },
                },
                b'&' => match self.look(1) {
                    Some(b'&') => { self.eat(2); Token::LAnd },
                    _          => { self.eat(1); Token::And },
                },
                b'|' => match self.look(1) {
                    Some(b'|') => { self.eat(2); Token::LOr },
                    _          => { self.eat(1); Token::Or },
                },
                b'-' => match self.look(1) {
                    Some(b'>') => { self.eat(2); Token::Arrow },
                    _          => { self.eat(1); Token::Sub },
                },
                b'/' => match self.look(1) {
                    Some(b'/') => {
                        let mut n = 2;
                        loop {
                            if let Some(b'\n') | None = self.look(n) {
                                n += 1;
                                break
                            }
                            n += 1;
                        }
                        self.eat(n);
                        continue;
                    },
                    _          => { self.eat(1); Token::Div },
                },
                b'~' => { self.eat(1); Token::Tilde },
                b'*' => { self.eat(1); Token::Mul },
                b'%' => { self.eat(1); Token::Rem },
                b'+' => { self.eat(1); Token::Add },
                b'^' => { self.eat(1); Token::Xor },
                b'(' => { self.eat(1); Token::LParen },
                b')' => { self.eat(1); Token::RParen },
                b'[' => { self.eat(1); Token::LSq },
                b']' => { self.eat(1); Token::RSq },
                b'{' => { self.eat(1); Token::LCurly },
                b'}' => { self.eat(1); Token::RCurly },
                b';' => { self.eat(1); Token::Semicolon },
                b':' => { self.eat(1); Token::Colon },
                b',' => { self.eat(1); Token::Comma },
                byte => { self.eat(1); Token::Error(byte) },
            };
        })
    }
}
