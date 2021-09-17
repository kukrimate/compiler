// SPDX-License-Identifier: GPL-2.0-only

//
// Recursive descent parser for the grammer described in "grammar.txt"
//

use super::ast::{Record,Type,Expr,Init,Stmt,Vis,Static,Func,File};
use super::lex::{Lexer,Token};

use std::collections::HashMap;
use std::rc::Rc;

//
// Parser
//

struct Parser<'source> {
    // Current file
    file: File,
    // Lexer and temorary token
    tmp: Option<Token>,
    lex: &'source mut Lexer<'source>,
    // Current string literal
    litno: usize,
}

macro_rules! want {
    ($self:expr, $pattern:pat, $err:expr) => {
        match $self.tmp {
            Some($pattern) => $self.tmp = $self.lex.next(),
            _ => panic!($err),
        }
    }
}

macro_rules! maybe_want {
    ($self:expr, $pattern:pat) => {
        match $self.tmp {
            Some($pattern) => {
                $self.tmp = $self.lex.next();
                true
            },
            _ => false,
        }
    }
}

impl<'source> Parser<'source> {
    fn new(lex: &'source mut Lexer<'source>) -> Parser<'source> {
        Parser {
            file: File::new(),
            tmp: lex.next(),
            lex: lex,
            litno: 0,
        }
    }

    fn next_token(&mut self) -> Token {
        std::mem::replace(&mut self.tmp, self.lex.next()).unwrap()
    }

    // FIXME: take string literal type into account
    fn make_string_lit(&mut self, _: Type, data: Rc<str>) -> Rc<str> {
        // Create globally unique name for the literal
        let name: Rc<str> = Rc::from(format!("_slit_{}", self.litno));
        self.litno += 1;

        // Create NUL-terminated initializer for the string
        let mut list = Vec::new();
        for b in data.as_bytes() {
            list.push(Init::Base(Expr::Const(Type::U8, *b as usize)));
        }
        list.push(Init::Base(Expr::Const(Type::U8, 0)));

        // Create static variable for it
        self.file.statics.insert(name.clone(),
            Static {
                vis: Vis::Private,
                name: name.clone(),
                dtype: Type::Array {
                    elem_count: list.len(),
                    elem_type: Box::from(Type::U8),
                },
                init: Some(Init::List(list)),
            });

        name
    }

    fn want_ident(&mut self) -> Rc<str> {
        match &self.tmp {
            Some(Token::Ident(_)) => {
                if let Token::Ident(s) = self.next_token() {
                    s
                } else {
                    unreachable!()
                }
            },
            tok @ _ => panic!("Expected identifier, got {:?}!", tok),
        }
    }

    fn want_label(&mut self) -> Rc<str> {
        match &self.tmp {
            Some(Token::Label(_)) => {
                if let Token::Label(s) = self.next_token() {
                    s
                } else {
                    unreachable!()
                }
            },
            tok @ _ => panic!("Expected label, got {:?}!", tok),
        }
    }

    fn maybe_want_vis(&mut self) -> Vis {
        if maybe_want!(self, Token::Export) {
            Vis::Export
        } else if maybe_want!(self, Token::Extern) {
            Vis::Extern
        } else {
            Vis::Private
        }
    }

    fn want_type_suffix(&mut self) -> Type {
        match self.next_token() {
            Token::U8   => Type::U8,
            Token::I8   => Type::I8,
            Token::U16  => Type::U16,
            Token::I16  => Type::I16,
            Token::U32  => Type::U32,
            Token::I32  => Type::I32,
            Token::U64  => Type::U64,
            Token::I64  => Type::I64,
            _ => panic!("Invalid type suffix!"),
        }
    }

    fn want_primary(&mut self) -> Expr {
        match self.next_token() {
            Token::LParen => {
                let expr = self.want_expr();
                want!(self, Token::RParen, "Missing )");
                expr
            },
            Token::Str(s) => {
                let string_type = self.want_type_suffix();
                // String literal becomes a pointer to a static
                Expr::Ref(Box::from(
                    Expr::Ident(self.make_string_lit(string_type, s))))
            },
            Token::Ident(s) => Expr::Ident(s),
            Token::Constant(val) => Expr::Const(self.want_type_suffix(), val),
            _ => panic!("Invalid constant value!"),
        }
    }

    fn want_postfix(&mut self) -> Expr {
        let mut expr = self.want_primary();
        loop {
            if maybe_want!(self, Token::Dot) {
                expr = Expr::Field(Box::from(expr), self.want_ident());
            } else if maybe_want!(self, Token::LParen) {
                let mut args = Vec::new();
                while !maybe_want!(self, Token::RParen) {
                    args.push(self.want_expr());
                    if !maybe_want!(self, Token::Comma) {
                        want!(self, Token::RParen, "Expected )");
                        break;
                    }
                }
                expr = Expr::Call(Box::from(expr), args);
            } else if maybe_want!(self, Token::LSq) {
                // NOTE: array indexing desugars into pointer arithmetic
                // FIXME: we should take the pointed type's size into account
                expr = Expr::Deref(Box::from(
                    Expr::Add(Box::from(expr), Box::from(self.want_expr()))));
                want!(self, Token::RSq, "Expected ]");
            } else {
                return expr;
            }
        }
    }

    fn want_unary(&mut self) -> Expr {
        if maybe_want!(self, Token::Sub) {
            self.want_unary().make_neg()
        } else if maybe_want!(self, Token::Tilde) {
            self.want_unary().make_inv()
        } else if maybe_want!(self, Token::Mul) {
            Expr::Deref(Box::from(self.want_unary()))
        } else if maybe_want!(self, Token::And) {
            Expr::Ref(Box::from(self.want_unary()))
        } else if maybe_want!(self, Token::Add) {
            self.want_unary()
        } else {
            self.want_postfix()
        }
    }

    fn want_cast(&mut self) -> Expr {
        let expr = self.want_unary();
        if maybe_want!(self, Token::Cast) {
            expr.make_cast(self.want_type())
        } else {
            expr
        }
    }

    fn want_mul(&mut self) -> Expr {
        let expr = self.want_cast();
        if maybe_want!(self, Token::Mul) {
            expr.make_mul(self.want_mul())
        } else if maybe_want!(self, Token::Div) {
            expr.make_div(self.want_mul())
        } else if maybe_want!(self, Token::Rem) {
            expr.make_rem(self.want_mul())
        } else {
            expr
        }
    }

    fn want_add(&mut self) -> Expr {
        let expr = self.want_mul();
        if maybe_want!(self, Token::Add) {
            expr.make_add(self.want_add())
        } else if maybe_want!(self, Token::Sub) {
            expr.make_sub(self.want_add())
        } else {
            expr
        }
    }

    fn want_shift(&mut self) -> Expr {
        let expr = self.want_add();
        if maybe_want!(self, Token::Lsh) {
            expr.make_lsh(self.want_shift())
        } else if maybe_want!(self, Token::Rsh) {
            expr.make_rsh(self.want_shift())
        } else {
            expr
        }
    }

    fn want_and(&mut self) -> Expr {
        let expr = self.want_shift();
        if maybe_want!(self, Token::And) {
            expr.make_and(self.want_and())
        } else {
            expr
        }
    }

    fn want_xor(&mut self) -> Expr {
        let expr = self.want_and();
        if maybe_want!(self, Token::Xor) {
            expr.make_xor(self.want_xor())
        } else {
            expr
        }
    }

    fn want_or(&mut self) -> Expr {
        let expr = self.want_xor();
        if maybe_want!(self, Token::Or) {
            expr.make_or(self.want_or())
        } else {
            expr
        }
    }

    fn want_expr(&mut self) -> Expr {
        self.want_or()
    }

    fn want_type(&mut self) -> Type {
        match self.next_token() {
            Token::U8   => Type::U8,
            Token::I8   => Type::I8,
            Token::U16  => Type::U16,
            Token::I16  => Type::I16,
            Token::U32  => Type::U32,
            Token::I32  => Type::I32,
            Token::U64  => Type::U64,
            Token::I64  => Type::I64,
            Token::Mul  => Type::Ptr { base_type: Box::new(self.want_type()) },
            Token::LSq  => {
                let elem_type = Box::new(self.want_type());
                want!(self, Token::Semicolon, "Expected ;");
                let elem_count_expr = self.want_expr();
                want!(self, Token::RSq, "Expected ]");
                Type::Array {
                    elem_type: elem_type,
                    elem_count: match elem_count_expr {
                        Expr::Const(_, val) => val,
                        _ => panic!("Array element count must be constnat"),
                    }
                }
            },
            Token::Ident(ref ident) => {
                if let Some(record) = self.file.records.get(ident) {
                    Type::Record(record.clone())
                } else {
                    panic!("Non-existent type {}", ident)
                }
            },
            _ => panic!("Invalid typename!"),
        }
    }

    fn want_record(&mut self, is_union: bool) -> Record {
        let mut fields = HashMap::new();
        let mut size = 0usize;
        // Read fields until }
        want!(self, Token::LCurly, "Expected left curly");
        while !maybe_want!(self, Token::RCurly) {
            let ident = self.want_ident();
            want!(self, Token::Colon, "Expected :");
            let r#type = self.want_type();
            let offset;
            let cur_size = r#type.get_size();
            if is_union {
                offset = 0;
                if size < cur_size {
                    size = cur_size;
                }
            } else {
                offset = size;
                size += cur_size;
            }
            fields.insert(ident, (r#type, offset));
            if !maybe_want!(self, Token::Comma) {
                want!(self, Token::RCurly, "Expected right curly");
                break;
            }
        }
        // Record type declaration must end in semicolon
        want!(self, Token::Semicolon, "Expected ;");

        Record {
            fields: fields,
            size: size,
        }
    }

    fn want_initializer(&mut self) -> Init {
        if maybe_want!(self, Token::LCurly) {
            let mut list = Vec::new();
            while !maybe_want!(self, Token::RCurly) {
                list.push(self.want_initializer());
                if !maybe_want!(self, Token::Comma) {
                    want!(self, Token::RCurly, "Expected right curly");
                    break;
                }
            }
            Init::List(list)
        } else {
            Init::Base(self.want_expr())
        }
    }

    fn want_stmt(&mut self) -> Stmt {
        let stmt = match self.next_token() {
            Token::Eval     => Stmt::Eval(self.want_expr()),
            Token::Ret      => {
                if maybe_want!(self, Token::Semicolon) {
                    // NOTE: return since we already have the semicolon
                    return Stmt::Ret(None)
                } else {
                    Stmt::Ret(Some(self.want_expr()))
                }
            },
            Token::Auto     => {
                let ident = self.want_ident();
                want!(self, Token::Colon, "Expected :");
                let r#type = self.want_type();
                let mut init = None;
                if maybe_want!(self, Token::Eq) {
                    init = Some(self.want_initializer());
                }
                Stmt::Auto(ident, r#type, init)
            },
            Token::Label(s) => Stmt::Label(s),
            Token::Set      => {
                let var = self.want_expr();
                want!(self, Token::Eq, "Expected =");
                Stmt::Set(var, self.want_expr())
            },
            Token::Jmp      => Stmt::Jmp(self.want_label()),
            Token::Jeq      => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jeq(label, expr1, self.want_expr())
            },
            Token::Jl       => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jl(label, expr1, self.want_expr())
            },
            Token::Jle      => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jle(label, expr1, self.want_expr())
            },
            Token::Jg       => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jg(label, expr1, self.want_expr())
            },
            Token::Jge      => {
                let label = self.want_label();
                want!(self, Token::Comma, "Expected ,");
                let expr1 = self.want_expr();
                want!(self, Token::Comma, "Expected ,");
                Stmt::Jge(label, expr1, self.want_expr())
            },
            tok @ _ => panic!("Invalid statement: {:?}", tok),
        };
        if let Stmt::Label(_) = stmt {
            want!(self, Token::Colon, "Expected :");
        } else {
            want!(self, Token::Semicolon, "Expected ;");
        }
        stmt
    }

    fn process(&mut self) {
        while !self.tmp.is_none() {
            match self.next_token() {
                Token::Record => {
                    let ident = self.want_ident();
                    let record = self.want_record(false);
                    self.file.records.insert(ident, Rc::from(record));
                },
                Token::Union => {
                    let ident = self.want_ident();
                    let union = self.want_record(true);
                    self.file.records.insert(ident, Rc::from(union));
                },
                Token::Static => {
                    let vis = self.maybe_want_vis();
                    let ident = self.want_ident();
                    want!(self, Token::Colon, "Expected :");
                    let dtype = self.want_type();
                    let mut init = None;
                    if maybe_want!(self, Token::Eq) {
                        init = Some(self.want_initializer());
                    }
                    want!(self, Token::Semicolon, "Expected ;");

                    self.file.statics.insert(ident.clone(), Static {
                        vis: vis,
                        name: ident,
                        dtype: dtype,
                        init: init
                    });
                },
                Token::Fn => {
                    let vis = self.maybe_want_vis();
                    let mut func = Func::new(vis, self.want_ident());

                    // Read parameters
                    want!(self, Token::LParen, "Expected (");
                    while !maybe_want!(self, Token::RParen) {
                        // Last parameter can be varargs
                        if maybe_want!(self, Token::Varargs) {
                            func.varargs = true;
                            want!(self, Token::RParen, "Expected )");
                            break;
                        }
                        // Otherwise try reading a normal parameter
                        let ident = self.want_ident();
                        want!(self, Token::Colon, "Expected :");
                        let r#type = self.want_type();
                        func.params.push((ident, r#type));
                        if !maybe_want!(self, Token::Comma) {
                            want!(self, Token::RParen, "Expected )");
                            break;
                        }
                    }

                    // Read return type (if any)
                    if maybe_want!(self, Token::Arrow) {
                        func.rettype = self.want_type();
                    }

                    // Read body (if present)
                    if !maybe_want!(self, Token::Semicolon) {
                        want!(self, Token::LCurly, "Expected left curly");
                        while !maybe_want!(self, Token::RCurly) {
                            func.stmts.push(self.want_stmt());
                        }
                    }

                    self.file.funcs.insert(func.name.clone(), func);
                },
                _ => panic!("Expected record, union, static or function!"),
            }
        }
    }
}

pub fn parse_file(data: &str) -> File {
    let mut lexer = Lexer::new(data);
    let mut parser = Parser::new(&mut lexer);
    parser.process();
    parser.file
}
