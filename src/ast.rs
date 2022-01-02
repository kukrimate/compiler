// SPDX-License-Identifier: GPL-2.0-only

//
// Recursive descent parser for the grammer described in "grammar.txt"
//

#![macro_use]

use crate::lex::{Lexer,Token};
use crate::gen::Gen;
use std::collections::HashMap;
use std::rc::Rc;

macro_rules! round_up {
    ($val:expr,$bound:expr) => {
        ($val + $bound - 1) / $bound * $bound
    }
}

//
// Type expression
//

#[derive(Clone,Debug,PartialEq)]
pub enum Type {
    Var(usize),

    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    USize,

    Bool,

    Ptr {
        // What type does this point to?
        base_type: Box<Type>,
    },

    Array {
        // Type of array elements
        elem_type: Box<Type>,
        // Number of array elements
        elem_count: Option<usize>,
    },

    // Product type
    Record {
        // Name of the record (this must match during type checking)
        name: Rc<str>,
        // Name lookup table
        lookup: HashMap<Rc<str>, usize>,
        // Field types and offsets (in declaration order)
        fields: Box<[(Type, usize)]>,
        // Pre-calculated alingment and size
        align: usize,
        size: usize,
    },

    // Return type from a procedure
    Void,

    // Function
    Func {
        params: Box<[Type]>,
        varargs: bool,
        rettype: Box<Type>,
    },
}

impl Type {
    pub fn get_align(&self) -> usize {
        match self {
            Type::U8 | Type::I8 | Type::Bool => 1,
            Type::U16 => 2,
            Type::I16 => 2,
            Type::U32 => 4,
            Type::I32 => 4,
            Type::U64 => 8,
            Type::I64 => 8,
            Type::USize => 8,
            Type::Ptr {..} => 8,
            Type::Array { elem_type, .. } => elem_type.get_align(),
            Type::Record { align, .. } => *align,
            Type::Var(_) | Type::Void | Type::Func {..}
                => unreachable!(),
        }
    }

    pub fn get_size(&self) -> usize {
        match self {
            Type::Bool | Type::U8 | Type::I8 => 1,
            Type::U16 => 2,
            Type::I16 => 2,
            Type::U32 => 4,
            Type::I32 => 4,
            Type::U64 => 8,
            Type::I64 => 8,
            Type::USize => 8,
            Type::Ptr {..} => 8,
            Type::Array { elem_type, elem_count }
                => elem_type.get_size() * elem_count
                        .expect("Array without element count allocated"),
            Type::Record { size, .. } => *size,
            Type::Var(_) | Type::Void | Type::Func {..}
                => unreachable!(),
        }
    }
}

//
// Abstract syntax tree elements
//

#[derive(Clone,Debug)]
pub enum ExprKind {
    // Constant value
    Const(usize),
    // Array/Record literal
    Compound(Vec<Expr>),
    // Reference to symbol
    Sym(Rc<str>),
    // Postfix expressions
    Field(Box<Expr>, usize),
    Call(Box<Expr>, Vec<Expr>),
    Elem(Box<Expr>, Box<Expr>),
    // Prefix expressions
    Ref(Box<Expr>),
    Deref(Box<Expr>),
    Not(Box<Expr>),
    LNot(Box<Expr>),
    Neg(Box<Expr>),
    // Cast expression
    Cast(Box<Expr>),
    // Binary expressions
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Rem(Box<Expr>, Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Lsh(Box<Expr>, Box<Expr>),
    Rsh(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Gt(Box<Expr>, Box<Expr>),
    Ge(Box<Expr>, Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Ne(Box<Expr>, Box<Expr>),
    LAnd(Box<Expr>, Box<Expr>),
    LOr(Box<Expr>, Box<Expr>),
}

#[derive(Clone,Debug)]
pub struct Expr {
    pub ty: Type,
    pub kind: ExprKind,
}

#[derive(Debug)]
pub enum Stmt {
    Block(Vec<Stmt>),
    Eval(Expr),
    Ret(Option<Expr>),
    Auto(Rc<str>, Type, Option<Expr>),
    Label(Rc<str>),
    Set(Expr, Expr),
    Jmp(Rc<str>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    While(Expr, Box<Stmt>),
}

#[derive(Debug,PartialEq)]
pub enum Vis {
    Private,    // Internal definition
    Export,     // Exported definition
    Extern,     // External definition reference
}

//
// Chained hash tables used for a symbol table
//

struct SymTab {
    list: Vec<HashMap<Rc<str>, Type>>,
}

impl SymTab {
    fn new() -> SymTab {
        let mut cm = SymTab {
            list: Vec::new(),
        };
        cm.list.push(HashMap::new());
        cm
    }

    fn insert(&mut self, name: Rc<str>, ty: Type) {
        if let Some(inner_scope) = self.list.last_mut() {
            if let Some(_) = inner_scope.insert(name.clone(), ty) {
                panic!("Re-declaration of {}", name)
            }
        } else {
            unreachable!();
        }
    }

    fn lookup(&mut self, name: &Rc<str>) -> &Type {
        for scope in self.list.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return ty;
            }
        }
        panic!("Unknown identifier {}", name)
    }

    fn push_scope(&mut self) {
        self.list.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        if self.list.len() < 2 {
            unreachable!();
        }
        self.list.pop();
    }
}

//
// Parser context
//

struct Parser<'source> {
    // Lexer and temorary token
    tmp: Option<Token>,
    lex: &'source mut Lexer<'source>,
    // Code generation backend
    gen: &'source mut Gen,
    // Currently defined record types
    records: HashMap<Rc<str>, Type>,
    // Symbol table
    symtab: SymTab,
    // Type variable index
    tvar: usize,
    // Type variable constraints
    tvarmap: HashMap<usize, Type>,
}

macro_rules! want {
    ($self:expr, $want:path) => {
        match $self.tmp {
            Some($want) => $self.tmp = $self.lex.next(),
            _ => panic!("Expected {:?} got {:?}", $want, $self.tmp),
        }
    }
}

macro_rules! maybe_want {
    ($self:expr, $want:pat) => {
        match $self.tmp {
            Some($want) => {
                $self.tmp = $self.lex.next();
                true
            },
            _ => false,
        }
    }
}

impl<'source> Parser<'source> {
    fn new(lex: &'source mut Lexer<'source>, gen: &'source mut Gen) -> Parser<'source> {
        Parser {
            tmp: lex.next(),
            lex: lex,
            gen: gen,
            records: HashMap::new(),
            symtab: SymTab::new(),
            tvar: 0,
            tvarmap: HashMap::new(),
        }
    }

    // Create a new unique type variable
    fn next_tvar(&mut self) -> Type {
        let tvar = self.tvar;
        self.tvar += 1;
        Type::Var(tvar)
    }

    // Unify two types
    fn unify(&mut self, ty1: Type, ty2: Type) -> Type {
        match (ty1, ty2) {
            // Recurse: aggregate types are composed of other types
            // NOTE: records and functions aren't treated as aggregates for now
            // as those form boundaries for type deduction and must be explictly
            // annotated
            (Type::Ptr { base_type: base_ty1 }, Type::Ptr { base_type: base_ty2 })
                => Type::Ptr {
                    base_type: Box::new(self.unify(*base_ty1, *base_ty2))
                },
            (Type::Array { elem_type: elem_ty1, elem_count: elem_cnt1 },
             Type::Array { elem_type: elem_ty2, elem_count: elem_cnt2 })
                => Type::Array {
                    elem_type: Box::new(self.unify(*elem_ty1, *elem_ty2)),
                    elem_count:
                        // Unknown length arrays come from array types created
                        // when indexing
                        match (elem_cnt1, elem_cnt2) {
                            (Some(cnt1), Some(cnt2))
                                => if cnt1 != cnt2 {
                                    panic!("Tried to unify arrays with different size")
                                } else {
                                    Some(cnt1)
                                },
                            (Some(cnt), None) | (None, Some(cnt))
                                => Some(cnt),
                            (None, None) => None,
                        }
                },

            // Base case type variables
            (Type::Var(var), newty) | (newty, Type::Var(var)) => {
                let ty = if let Some(prevty) = self.tvarmap.remove(&var) {
                    // If there was a previous bound for this variable,
                    // unify them and insert the unified type
                    self.unify(prevty, newty)
                } else {
                    // If not, insert the new type into the table
                    newty
                };
                self.tvarmap.insert(var, ty.clone());
                ty
            },

            // Base case: basic types can be compared directly
            (ty1, ty2) => {
                if ty1 != ty2 {
                    panic!("Incompatible types {:?}, {:?}", ty1, ty2)
                }
                ty1
            }
        }
    }

    // Find the literal type for a type expression that might contain type variables
    // NOTE: only works after all constraints were unified
    fn lit_type(&mut self, ty: Type) -> Type {
        match ty {
            // Recurse: aggregate types are composed of other types
            // NOTE: records and functions aren't treated as aggregates for now
            // as those form boundaries for type deduction and must be explictly
            // annotated
            Type::Ptr { base_type }
                => Type::Ptr {
                    base_type: Box::new(self.lit_type(*base_type))
                },
            Type::Array { elem_type, elem_count }
                => Type::Array {
                    elem_type: Box::new(self.lit_type(*elem_type)),
                    elem_count: elem_count
                },

            // Base case: type variables
            Type::Var(var)
                => if let Some(ty) = self.tvarmap.get(&var).cloned() {
                    self.lit_type(ty)
                } else {
                    panic!("Type variable {} not deducible, provide more information", var);
                },

            // Base case: basic types
            ty => ty,
        }
    }

    //
    // Expression AST constructors
    //
    // These run when a function is first parsed, thus not all types will be
    // deduced here. Hence these functions should only type check something
    // where an expression by definition must have its type deducible from
    // previous context.
    //

    fn make_const(&mut self, ty: Type, val: usize) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Const(val)
        }
    }

    fn make_sym(&mut self, ty: Type, name: Rc<str>) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Sym(name)
        }
    }

    fn make_field(&mut self, record: Expr, name: Rc<str>) -> Expr {
        // Records are type deduction boundaries,
        // we can finish deducing here and enforce.
        if let Type::Record { lookup, fields, .. } = self.lit_type(record.ty.clone()) {
            let (ty, off) = if let Some(idx) = lookup.get(&name) {
                fields[*idx].clone()
            } else {
                panic!("Unknown field {}", name);
            };
            Expr {
                ty: ty,
                kind: ExprKind::Field(Box::new(record), off)
            }
        } else {
            panic!("Dot operator on non-record type");
        }
    }

    fn make_call(&mut self, func: Expr, args: Vec<Expr>) -> Expr {
        // Function calls are type deduction boundaries,
        // we can finish deducing here and enforce.
        if let Type::Func { params, varargs, rettype } = self.lit_type(func.ty.clone()) {
            // Make sure the number of arguments is correct
            if args.len() < params.len() {
                panic!("Too few arguments");
            }
            if !varargs && args.len() > params.len() {
                panic!("Too many arguments");
            }

            // For each non-varargs argument, unify the argument
            // expression's type with the function parameter's type.
            for (param_ty, arg) in params.iter().zip(args.iter()) {
                self.unify(param_ty.clone(), arg.ty.clone());
            }

            Expr {
                ty: *rettype.clone(),
                kind: ExprKind::Call(Box::new(func), args)
            }
        } else {
            panic!("() operator on non-function type");
        }
    }

    fn make_elem(&mut self, array: Expr, index: Expr) -> Expr {
        // The type we expect is an array of something, create such a type
        // with a type variable as the element type, and unify it with the
        // arrray expression's type. This will ensure that we will have a type
        // as required.
        let elem_ty = self.next_tvar();
        self.unify(array.ty.clone(), Type::Array {
            elem_type: Box::new(elem_ty.clone()),
            elem_count: None,
        });
        self.unify(index.ty.clone(), Type::USize);
        Expr {
            ty: elem_ty,
            kind: ExprKind::Elem(Box::new(array), Box::new(index))
        }
    }


    fn make_ref(&mut self, expr: Expr) -> Expr {
        Expr {
            ty: Type::Ptr { base_type: Box::new(expr.ty.clone()) },
            kind: ExprKind::Ref(Box::new(expr))
        }
    }

    fn make_deref(&mut self, ptr: Expr) -> Expr {
        // We are doing the same thing here as in make_elem() above
        let base_ty = self.next_tvar();
        self.unify(ptr.ty.clone(), Type::Ptr {
            base_type: Box::new(base_ty.clone())
        });
        Expr {
            ty: base_ty,
            kind: ExprKind::Deref(Box::new(ptr))
        }
    }

    fn make_not(&mut self, expr: Expr) -> Expr {
        Expr {
            ty: expr.ty.clone(),
            kind: ExprKind::Not(Box::new(expr))
        }
    }

    fn make_lnot(&mut self, expr: Expr) -> Expr {
        Expr {
            ty: expr.ty.clone(),
            kind: ExprKind::LNot(Box::new(expr))
        }
    }

    fn make_neg(&mut self, expr: Expr) -> Expr {
        Expr {
            ty: expr.ty.clone(),
            kind: ExprKind::Neg(Box::new(expr))
        }
    }

    fn make_cast(&mut self, expr: Expr, ty: Type) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Cast(Box::new(expr))
        }
    }

    fn make_mul(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(expr.ty.clone(), expr2.ty.clone()),
            kind: ExprKind::Mul(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_div(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(expr.ty.clone(), expr2.ty.clone()),
            kind: ExprKind::Div(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_rem(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(expr.ty.clone(), expr2.ty.clone()),
            kind: ExprKind::Rem(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_add(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(expr.ty.clone(), expr2.ty.clone()),
            kind: ExprKind::Add(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_sub(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(expr.ty.clone(), expr2.ty.clone()),
            kind: ExprKind::Sub(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_lsh(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: expr.ty.clone(),
            kind: ExprKind::Lsh(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_rsh(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: expr.ty.clone(),
            kind: ExprKind::Rsh(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_lt(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(expr.ty.clone(), expr2.ty.clone());
        Expr {
            ty: Type::Bool,
            kind: ExprKind::Lt(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_le(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(expr.ty.clone(), expr2.ty.clone());
        Expr {
            ty: Type::Bool,
            kind: ExprKind::Le(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_gt(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(expr.ty.clone(), expr2.ty.clone());
        Expr {
            ty: Type::Bool,
            kind: ExprKind::Gt(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_ge(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(expr.ty.clone(), expr2.ty.clone());
        Expr {
            ty: Type::Bool,
            kind: ExprKind::Ge(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_eq(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(expr.ty.clone(), expr2.ty.clone());
        Expr {
            ty: Type::Bool,
            kind: ExprKind::Eq(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_ne(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(expr.ty.clone(), expr2.ty.clone());
        Expr {
            ty: Type::Bool,
            kind: ExprKind::Ne(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_and(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(expr.ty.clone(), expr2.ty.clone()),
            kind: ExprKind::And(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_xor(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(expr.ty.clone(), expr2.ty.clone()),
            kind: ExprKind::Xor(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_or(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(expr.ty.clone(), expr2.ty.clone()),
            kind: ExprKind::Or(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_land(&mut self, expr: Expr, expr2: Expr) -> Expr {
        // Mostly pointless, but we can add a constraint on these being bool
        // to the deducer. This should not matter for valid code, but for example
        // this will force un-initialized lets used in this context to be deduced
        // to be bool.
        self.unify(Type::Bool, expr.ty.clone());
        self.unify(Type::Bool, expr2.ty.clone());
        Expr {
            ty: Type::Bool,
            kind: ExprKind::LAnd(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_lor(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(Type::Bool, expr.ty.clone());
        self.unify(Type::Bool, expr2.ty.clone());
        Expr {
            ty: Type::Bool,
            kind: ExprKind::LOr(Box::new(expr), Box::new(expr2))
        }
    }

    //
    // Finalize the AST before feeding it to the code generator.
    // Right now this does the following things:
    //  1. Replace type variables with deduced literal types
    //  2. Do type checking on expressions that support type deduction
    //     from later context (everything but field access and function calls)
    //  3. Fold constant expressions into a single constant AST node
    //

    fn finalize_expr(&mut self, mut expr: Expr) -> Expr {
        // Make sure the expression type is in its literal form
        expr.ty = self.lit_type(expr.ty);

        // The following steps depend on expression kind
        expr.kind = match expr.kind {
            kind @ (ExprKind::Const(_) | ExprKind::Sym(_)) => kind,
            ExprKind::Compound(exprs)
                => ExprKind::Compound(exprs.into_iter()
                    .map(|expr| self.finalize_expr(expr)).collect()),

            ExprKind::Field(record, off)
                => ExprKind::Field(Box::new(self.finalize_expr(*record)), off),
            ExprKind::Call(func, args)
                => ExprKind::Call(Box::new(self.finalize_expr(*func)),
                    args.into_iter().map(|expr| self.finalize_expr(expr)).collect()),
            ExprKind::Elem(array, index)
                => ExprKind::Elem(
                    Box::new(self.finalize_expr(*array)),
                    Box::new(self.finalize_expr(*index))),

            ExprKind::Ref(inner)
                => ExprKind::Ref(Box::new(self.finalize_expr(*inner))),
            ExprKind::Deref(inner)
                => ExprKind::Deref(Box::new(self.finalize_expr(*inner))),
            ExprKind::Not(inner)
                => ExprKind::Not(Box::new(self.finalize_expr(*inner))),
            ExprKind::LNot(inner)
                => ExprKind::LNot(Box::new(self.finalize_expr(*inner))),
            ExprKind::Neg(inner)
                => ExprKind::Neg(Box::new(self.finalize_expr(*inner))),
            ExprKind::Cast(inner)
                => ExprKind::Cast(Box::new(self.finalize_expr(*inner))),

            ExprKind::Mul(lhs, rhs)
                => ExprKind::Mul(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Div(lhs, rhs)
                => ExprKind::Div(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Rem(lhs, rhs)
                => ExprKind::Rem(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Add(lhs, rhs)
                => ExprKind::Add(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Sub(lhs, rhs)
                => ExprKind::Sub(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Lsh(lhs, rhs)
                => ExprKind::Lsh(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Rsh(lhs, rhs)
                => ExprKind::Rsh(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::And(lhs, rhs)
                => ExprKind::And(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Xor(lhs, rhs)
                => ExprKind::Xor(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Or(lhs, rhs)
                => ExprKind::Or(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Lt(lhs, rhs)
                => ExprKind::Lt(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Le(lhs, rhs)
                => ExprKind::Le(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Gt(lhs, rhs)
                => ExprKind::Gt(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Ge(lhs, rhs)
                => ExprKind::Ge(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Eq(lhs, rhs)
                => ExprKind::Eq(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::Ne(lhs, rhs)
                => ExprKind::Ne(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::LAnd(lhs, rhs)
                => ExprKind::LAnd(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
            ExprKind::LOr(lhs, rhs)
                => ExprKind::LOr(Box::new(self.finalize_expr(*lhs)),
                    Box::new(self.finalize_expr(*rhs))),
        };

        // Finally return the expression
        expr
    }

    fn finalize_stmt(&mut self, stmt: Stmt) -> Stmt {
        match stmt {
            Stmt::Block(stmts) =>
                Stmt::Block(stmts.into_iter()
                    .map(|stmt| self.finalize_stmt(stmt)).collect()),
            Stmt::Eval(expr) =>
                Stmt::Eval(self.finalize_expr(expr)),
            Stmt::Ret(opt_expr) =>
                if let Some(expr) = opt_expr {
                    Stmt::Ret(Some(self.finalize_expr(expr)))
                } else {
                    Stmt::Ret(None)
                },
            Stmt::Auto(name, mut ty, mut opt_expr) => {
                // Obtain literal type for declaration
                ty = self.lit_type(ty);
                // Finalize initializer if present
                if let Some(expr) = opt_expr {
                    opt_expr = Some(self.finalize_expr(expr));
                }
                Stmt::Auto(name, ty, opt_expr)
            },
            Stmt::Set(dst, src)
                => Stmt::Set(self.finalize_expr(dst), self.finalize_expr(src)),
            Stmt::If(mut cond, mut then, mut opt_else) => {
                cond = self.finalize_expr(cond);
                then = Box::new(self.finalize_stmt(*then));
                if let Some(_else) = opt_else {
                    opt_else = Some(Box::new(self.finalize_stmt(*_else)));
                }
                Stmt::If(cond, then, opt_else)
            },
            Stmt::While(mut cond, mut body) => {
                cond = self.finalize_expr(cond);
                body = Box::new(self.finalize_stmt(*body));
                Stmt::While(cond, body)
            },
            stmt @ (Stmt::Label(_) | Stmt::Jmp(_)) => stmt,
        }
    }

    // Read the next token
    fn next_token(&mut self) -> Token {
        std::mem::replace(&mut self.tmp, self.lex.next()).unwrap()
    }

    // Read an identifier
    fn want_ident(&mut self) -> Rc<str> {
        match &self.tmp {
            Some(Token::Ident(_)) => {
                if let Token::Ident(s) = self.next_token() {
                    s
                } else {
                    unreachable!()
                }
            },
            tok => panic!("Expected identifier, got {:?}!", tok),
        }
    }

    // Read a visibility (or return Vis::Private)
    fn maybe_want_vis(&mut self) -> Vis {
        if maybe_want!(self, Token::Export) {
            Vis::Export
        } else if maybe_want!(self, Token::Extern) {
            Vis::Extern
        } else {
            Vis::Private
        }
    }

    // Read an integer type (or return None)
    fn want_type_suffix(&mut self) -> Option<Type> {
        // Match for type suffix
        let suf = match self.tmp {
            Some(Token::U8)   => Type::U8,
            Some(Token::I8)   => Type::I8,
            Some(Token::U16)  => Type::U16,
            Some(Token::I16)  => Type::I16,
            Some(Token::U32)  => Type::U32,
            Some(Token::I32)  => Type::I32,
            Some(Token::U64)  => Type::U64,
            Some(Token::I64)  => Type::I64,
            Some(Token::USize)=> Type::USize,
            _ => return None,
        };
        // Replace temporary token if matched
        self.tmp = self.lex.next();
        Some(suf)
    }

    fn want_array_literal(&mut self) -> Expr {
        let mut elem_ty = self.next_tvar();
        let mut elems = Vec::new();

        while !maybe_want!(self, Token::RSq) {
            let expr = self.want_expr();
            elem_ty = self.unify(elem_ty, expr.ty.clone());
            elems.push(expr);
            if !maybe_want!(self, Token::Comma) {
                want!(self, Token::RSq);
                break;
            }
        }

        Expr {
            ty: Type::Array {
                elem_type: Box::new(elem_ty),
                elem_count: Some(elems.len()),
            },
            kind: ExprKind::Compound(elems)
        }
    }

    fn want_record_literal(&mut self, tyname: Rc<str>) -> Expr {
        // Find record type
        let ty = if let Some(ty) = self.records.get(&tyname) {
            ty.clone()
        } else {
            panic!("Unknown record {}", tyname)
        };

        let (lookup, fields) = if let Type::Record { lookup, fields, .. } = &ty {
            (lookup, fields)
        } else {
            unreachable!()
        };

        // Read expressions
        let mut expr_map = HashMap::new();
        while !maybe_want!(self, Token::RCurly) {
            // Read field name
            let name = self.want_ident();

            // Check for duplicate field
            if let Some(_) = expr_map.get(&name) {
                panic!("Duplicate field {}", name);
            }

            // Read field initializer
            want!(self, Token::Colon);
            expr_map.insert(name, self.want_expr());

            // See if we've reached the end
            if !maybe_want!(self, Token::Comma) {
                want!(self, Token::RCurly);
                break;
            }
        }

        // Map expressions to fields
        let mut expr_vec = Vec::new();
        for (name, idx) in lookup {
            let ty = fields[*idx].0.clone();
            if let Some(expr) = expr_map.remove(name) {
                self.unify(ty, expr.ty.clone());
                expr_vec.push(expr);
            } else {
                panic!("Missing field {}", name);
            }
        }

        // Make sure there are no expressions left
        for name in expr_map.keys() {
            panic!("Unknown field {}", name);
        }

        Expr {
            ty: ty,
            kind: ExprKind::Compound(expr_vec)
        }
    }

    fn want_primary(&mut self) -> Expr {
        match self.next_token() {
            Token::LParen => {
                let expr = self.want_expr();
                want!(self, Token::RParen);
                expr
            },
            Token::Str(data) => {
                let chty = self.want_type_suffix().unwrap_or(Type::U8);
                let (name, ty) = self.gen.do_string(chty.clone(), &*data);
                // Replace string literal with reference to internal symbol
                let sym_expr = self.make_sym(ty, name);
                let mut ref_expr = self.make_ref(sym_expr);
                // HACK: this reference doesn't actually have an array pointer's
                // type, because C APIs degrade arrays to pointers to the first
                // element
                ref_expr.ty = Type::Ptr { base_type: Box::new(chty) };
                ref_expr
            },
            Token::Ident(name)
                => if maybe_want!(self, Token::LCurly) {
                    self.want_record_literal(name)
                } else {
                    let ty = self.symtab.lookup(&name).clone();
                    self.make_sym(ty, name)
                },
            Token::LSq
                => self.want_array_literal(),
            Token::Constant(val) => {
                let ty = if let Some(ty) = self.want_type_suffix() {
                    ty
                } else {
                    self.next_tvar()
                };
                self.make_const(ty, val)
            },
            Token::True
                => self.make_const(Type::Bool, 1),
            Token::False
                => self.make_const(Type::Bool, 0),
            _ => panic!("Invalid constant value!"),
        }
    }

    fn want_postfix(&mut self) -> Expr {
        let mut expr = self.want_primary();
        loop {
            if maybe_want!(self, Token::Dot) {
                let name = self.want_ident();
                expr = self.make_field(expr, name);
            } else if maybe_want!(self, Token::LParen) {
                let mut args = Vec::new();
                while !maybe_want!(self, Token::RParen) {
                    args.push(self.want_expr());
                    if !maybe_want!(self, Token::Comma) {
                        want!(self, Token::RParen);
                        break;
                    }
                }
                expr = self.make_call(expr, args);
            } else if maybe_want!(self, Token::LSq) {
                let index = self.want_expr();
                expr = self.make_elem(expr, index);
                want!(self, Token::RSq);
            } else {
                return expr;
            }
        }
    }

    fn want_unary(&mut self) -> Expr {
        if maybe_want!(self, Token::Sub) {
            let expr = self.want_unary();
            self.make_neg(expr)
        } else if maybe_want!(self, Token::Tilde) {
            let expr = self.want_unary();
            self.make_not(expr)
        } else if maybe_want!(self, Token::Excl) {
            let expr = self.want_unary();
            self.make_lnot(expr)
        } else if maybe_want!(self, Token::Mul) {
            let expr = self.want_unary();
            self.make_deref(expr)
        } else if maybe_want!(self, Token::And) {
            let expr = self.want_unary();
            self.make_ref(expr)
        } else if maybe_want!(self, Token::Add) {
            self.want_unary()
        } else {
            self.want_postfix()
        }
    }

    fn want_cast(&mut self) -> Expr {
        let mut expr = self.want_unary();
        while maybe_want!(self, Token::As) {
            let ty = self.want_type();
            expr = self.make_cast(expr, ty);
        }
        expr
    }

    fn want_mul(&mut self) -> Expr {
        let mut expr = self.want_cast();
        loop {
            if maybe_want!(self, Token::Mul) {
                let expr2 = self.want_cast();
                expr = self.make_mul(expr, expr2);
            } else if maybe_want!(self, Token::Div) {
                let expr2 = self.want_cast();
                expr = self.make_div(expr, expr2);
            } else if maybe_want!(self, Token::Rem) {
                let expr2 = self.want_cast();
                expr = self.make_rem(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_add(&mut self) -> Expr {
        let mut expr = self.want_mul();
        loop {
            if maybe_want!(self, Token::Add) {
                let expr2 = self.want_mul();
                expr = self.make_add(expr, expr2);
            } else if maybe_want!(self, Token::Sub) {
                let expr2 = self.want_mul();
                expr = self.make_sub(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_shift(&mut self) -> Expr {
        let mut expr = self.want_add();
        loop {
            if maybe_want!(self, Token::Lsh) {
                let expr2 = self.want_add();
                expr = self.make_lsh(expr, expr2);
            } else if maybe_want!(self, Token::Rsh) {
                let expr2 = self.want_add();
                expr = self.make_rsh(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_and(&mut self) -> Expr {
        let mut expr = self.want_shift();
        loop {
            if maybe_want!(self, Token::And) {
                let expr2 = self.want_shift();
                expr = self.make_and(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_xor(&mut self) -> Expr {
        let mut expr = self.want_and();
        loop {
            if maybe_want!(self, Token::Xor) {
                let expr2 = self.want_and();
                expr = self.make_xor(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_or(&mut self) -> Expr {
        let mut expr = self.want_xor();
        loop {
            if maybe_want!(self, Token::Or) {
                let expr2 = self.want_xor();
                expr = self.make_or(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_cmp(&mut self) -> Expr {
        let expr = self.want_or();
        if maybe_want!(self, Token::Lt) {
            let expr2 = self.want_or();
            self.make_lt(expr, expr2)
        } else if maybe_want!(self, Token::Le) {
            let expr2 = self.want_or();
            self.make_le(expr, expr2)
        } else if maybe_want!(self, Token::Gt) {
            let expr2 = self.want_or();
            self.make_gt(expr, expr2)
        } else if maybe_want!(self, Token::Ge) {
            let expr2 = self.want_or();
            self.make_ge(expr, expr2)
        } else if maybe_want!(self, Token::Eq) {
            let expr2 = self.want_or();
            self.make_eq(expr, expr2)
        } else if maybe_want!(self, Token::Ne) {
            let expr2 = self.want_or();
            self.make_ne(expr, expr2)
        } else {
            expr
        }
    }

    fn want_land(&mut self) -> Expr {
        let mut expr = self.want_cmp();
        loop {
            if maybe_want!(self, Token::LAnd) {
                let expr2 = self.want_cmp();
                expr = self.make_land(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_lor(&mut self) -> Expr {
        let mut expr = self.want_land();
        loop {
            if maybe_want!(self, Token::LAnd) {
                let expr2 = self.want_land();
                expr = self.make_lor(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_expr(&mut self) -> Expr {
        self.want_lor()
    }

    fn want_type(&mut self) -> Type {
        match self.next_token() {
            Token::Bool => Type::Bool,
            Token::U8   => Type::U8,
            Token::I8   => Type::I8,
            Token::U16  => Type::U16,
            Token::I16  => Type::I16,
            Token::U32  => Type::U32,
            Token::I32  => Type::I32,
            Token::U64  => Type::U64,
            Token::I64  => Type::I64,
            Token::USize=> Type::USize,
            Token::Mul  => Type::Ptr {
                base_type: Box::new(self.want_type())
            },
            Token::LSq  => {
                let elem_type = self.want_type();
                want!(self, Token::Semicolon);
                let expr = self.want_expr();
                self.unify(Type::USize, expr.ty.clone());
                let elem_count = match self.finalize_expr(expr).kind {
                    ExprKind::Const(val) => val,
                    _ => panic!("Array element count must be constnat"),
                };
                want!(self, Token::RSq);

                Type::Array {
                    elem_type: Box::new(elem_type),
                    elem_count: Some(elem_count)
                }
            },
            Token::Ident(name) => {
                if let Some(record) = self.records.get(&name) {
                    return record.clone();
                } else {
                    panic!("Non-existent type {}", name)
                }
            },
            Token::Fn => {
                // Read parameter types
                want!(self, Token::LParen);
                let mut params = Vec::new();
                let mut varargs = false;
                while !maybe_want!(self, Token::RParen) {
                    if maybe_want!(self, Token::Varargs) {
                        varargs = true;
                        want!(self, Token::LParen);
                        break;
                    }
                    params.push(self.want_type());
                    if !maybe_want!(self, Token::Comma) {
                        want!(self, Token::RParen);
                        break;
                    }
                }

                // Read return type
                let rettype = if maybe_want!(self, Token::Arrow) {
                    self.want_type()
                } else {
                    Type::Void
                };

                Type::Func {
                    params: params.into(),
                    varargs: varargs,
                    rettype: Box::new(rettype)
                }
            },
            _ => panic!("Invalid typename!"),
        }
    }

    fn want_record(&mut self, name: Rc<str>) -> Type {
        let mut lookup = HashMap::new();
        let mut fields = Vec::new();
        let mut max_align = 0;
        let mut offset = 0;

        want!(self, Token::LCurly);

        // Read fields until }
        while !maybe_want!(self, Token::RCurly) {
            lookup.insert(self.want_ident(), fields.len());
            want!(self, Token::Colon);

            let field_type = self.want_type();
            let field_align = field_type.get_align();
            let field_size = field_type.get_size();

            // Save maximum alignment of all fields
            if max_align < field_align {
                max_align = field_align;
            }

            // Round field offset to correct alignment
            offset = round_up!(offset, field_align);
            fields.push((field_type, offset));
            offset += field_size;

            if !maybe_want!(self, Token::Comma) {
                want!(self, Token::RCurly);
                break;
            }
        }

        // Record type declaration must end in semicolon
        want!(self, Token::Semicolon);

        Type::Record {
            name: name,
            lookup: lookup,
            fields: fields.into_boxed_slice(),
            align: max_align,
            // Round struct size to a multiple of it's alignment, this is needed
            // if an array is ever created of this struct
            size: round_up!(offset, max_align),
        }
    }

    fn want_stmts(&mut self, rettype: &Type) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        while !maybe_want!(self, Token::RCurly) {
            stmts.push(self.want_stmt(rettype));
        }
        stmts
    }

    fn want_block(&mut self, rettype: &Type) -> Stmt {
        self.symtab.push_scope();
        let stmts = self.want_stmts(rettype);
        self.symtab.pop_scope();
        Stmt::Block(stmts)
    }

    fn want_if(&mut self, rettype: &Type) -> Stmt {
        // Read conditional
        want!(self, Token::LParen);
        let cond = self.want_expr();
        want!(self, Token::RParen);

        // Read true branch
        want!(self, Token::LCurly);
        let then = self.want_block(rettype);

        // Read else branch if present
        let _else = if maybe_want!(self, Token::Else) {
            Some(Box::new(match self.next_token() {
                Token::LCurly => self.want_block(rettype),
                Token::If => self.want_if(rettype),
                _ => panic!("Expected block or if after else")
            }))
        } else {
            None
        };

        Stmt::If(cond, Box::new(then), _else)
    }

    fn want_while(&mut self, rettype: &Type) -> Stmt {
        // Read conditional
        want!(self, Token::LParen);
        let cond = self.want_expr();
        want!(self, Token::RParen);

        // Read body
        want!(self, Token::LCurly);
        let body = self.want_block(rettype);

        Stmt::While(cond, Box::new(body))
    }

    fn want_stmt(&mut self, rettype: &Type) -> Stmt {
        match self.next_token() {
            Token::LCurly => self.want_block(rettype),
            Token::Eval => {
                let stmt = Stmt::Eval(self.want_expr());
                want!(self, Token::Semicolon);
                stmt
            },
            Token::Ret => {
                let stmt = if maybe_want!(self, Token::Semicolon) {
                    // NOTE: return since we already have the semicolon
                    return Stmt::Ret(None)
                } else {
                    let expr = self.want_expr();
                    self.unify(expr.ty.clone(), rettype.clone());
                    Stmt::Ret(Some(expr))
                };
                want!(self, Token::Semicolon);
                stmt
            },
            Token::Auto => {
                let name = self.want_ident();

                // Read type (or create type variable)
                let mut ty = if maybe_want!(self, Token::Colon) {
                    self.want_type()
                } else {
                    self.next_tvar()
                };

                // Unify type with initializer type
                let expr = if maybe_want!(self, Token::Assign) {
                    let expr = self.want_expr();
                    ty = self.unify(ty, expr.ty.clone());
                    Some(expr)
                } else {
                    None
                };

                // Insert symbol
                self.symtab.insert(name.clone(), ty.clone());

                // Create statement
                let stmt = Stmt::Auto(name, ty, expr);
                want!(self, Token::Semicolon);
                stmt
            },
            Token::Ident(s) => {
                want!(self, Token::Colon);
                Stmt::Label(s)
            },
            Token::Set => {
                let dst = self.want_expr();
                want!(self, Token::Assign);
                let src = self.want_expr();
                want!(self, Token::Semicolon);

                self.unify(dst.ty.clone(), src.ty.clone());
                Stmt::Set(dst, src)
            },
            Token::Jmp => {
                let stmt = Stmt::Jmp(self.want_ident());
                want!(self, Token::Semicolon);
                stmt
            },
            Token::If => self.want_if(rettype),
            Token::While => self.want_while(rettype),
            tok => panic!("Invalid statement {:?}", tok),
        }
    }

    fn process(&mut self) {
        while !self.tmp.is_none() {
            match self.next_token() {
                Token::Record => {
                    let name = self.want_ident();
                    let record = self.want_record(name.clone());
                    self.records.insert(name, record);
                },
                Token::Static => {
                    let vis = self.maybe_want_vis();
                    let name = self.want_ident();

                    let mut ty = if maybe_want!(self, Token::Colon) {
                        self.want_type()
                    } else {
                        self.next_tvar()
                    };

                    if maybe_want!(self, Token::Assign) {
                        let mut expr = self.want_expr();
                        ty = self.unify(ty, expr.ty.clone());
                        expr = self.finalize_expr(expr);

                        self.gen.do_static_init(vis, name.clone(), ty.clone(), expr);
                        self.symtab.insert(name, ty);
                    } else {
                        self.gen.do_static(vis, name.clone(), ty.clone());
                        self.symtab.insert(name, ty);
                    }

                    want!(self, Token::Semicolon);
                },
                Token::Fn => {
                    let vis = self.maybe_want_vis();
                    let name = self.want_ident();

                    let mut params = Vec::new();
                    let mut varargs = false;

                    let mut param_tab = Vec::new();

                    // Read parameters
                    want!(self, Token::LParen);
                    while !maybe_want!(self, Token::RParen) {
                        // Last parameter can be varargs
                        if maybe_want!(self, Token::Varargs) {
                            varargs = true;
                            want!(self, Token::RParen);
                            break;
                        }
                        // Otherwise try reading a normal parameter
                        let param_name = self.want_ident();
                        want!(self, Token::Colon);
                        let param_type = self.want_type();
                        params.push(param_type.clone());
                        param_tab.push((param_name, param_type));
                        if !maybe_want!(self, Token::Comma) {
                            want!(self, Token::RParen);
                            break;
                        }
                    }

                    // Read return type (or set to void)
                    let rettype = if maybe_want!(self, Token::Arrow) {
                        self.want_type()
                    } else {
                        Type::Void
                    };

                    // Create symbol for function
                    let ty = Type::Func {
                        params: params.into(),
                        varargs: varargs,
                        rettype: Box::new(rettype.clone()),
                    };
                    self.symtab.insert(name.clone(), ty.clone());
                    self.gen.do_sym(vis, name.clone(), ty);

                    // Read body (if present)
                    if maybe_want!(self, Token::LCurly) {
                        self.symtab.push_scope();

                        // Add paramters to scope
                        for (name, ty) in &param_tab {
                            self.symtab.insert(name.clone(), ty.clone());
                        }

                        // Read body
                        let mut stmts = self.want_stmts(&rettype);

                        self.symtab.pop_scope();

                        // Generate body
                        println!("AST: {:#?}", stmts);
                        stmts = stmts.into_iter()
                            .map(|stmt| self.finalize_stmt(stmt)).collect();
                        println!("Constraints: {:#?}", self.tvarmap);
                        println!("Deduced AST: {:#?}", stmts);

                        self.gen.do_func(name, rettype, param_tab, stmts);
                    } else {
                        want!(self, Token::Semicolon)
                    }
                },
                _ => panic!("Expected record, union, static or function!"),
            }
        }
    }
}

pub fn parse_file(data: &str, gen: &mut Gen) {
    let mut lexer = Lexer::new(data);
    let mut parser = Parser::new(&mut lexer, gen);
    parser.process();
}
