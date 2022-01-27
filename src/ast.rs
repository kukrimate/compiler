// SPDX-License-Identifier: GPL-2.0-only

//
// Recursive descent parser for the grammer described in "grammar.txt"
//

use crate::lex::{Lexer,Token};
use crate::gen::Gen;
use crate::gen::Local;
use crate::util::PeekIter;
use std::collections::HashMap;
use std::rc::Rc;

macro_rules! want {
    ($self:expr, $want:path) => {
        match $self.next() {
            Some($want) => (),
            got => panic!("Expected {:?} got {:?}", $want, got),
        }
    }
}

macro_rules! maybe_want {
    ($self:expr, $want:pat) => {
        match $self.peek(0) {
            Some($want) => {
                $self.next();
                true
            },
            _ => false,
        }
    }
}

//
// Type expressions
//

#[derive(Clone,Debug)]
pub enum Ty {
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
        base_type: Box<Ty>,
    },

    Array {
        // Ty of array elements
        elem_type: Box<Ty>,
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
        fields: Box<[(Ty, usize)]>,
        // Pre-calculated alingment and size
        align: usize,
        size: usize,
    },

    // Return type from a procedure
    Void,

    // Function
    Func {
        params: Box<[Ty]>,
        varargs: bool,
        rettype: Box<Ty>,
    },
}

impl Ty {
    pub fn is_signed(&self) -> bool {
        match self {
            Ty::I8|Ty::I16|Ty::I32|Ty::I64 => true,
            _ => false
        }
    }

    pub fn get_align(&self) -> usize {
        match self {
            Ty::U8 | Ty::I8 | Ty::Bool => 1,
            Ty::U16 => 2,
            Ty::I16 => 2,
            Ty::U32 => 4,
            Ty::I32 => 4,
            Ty::U64 => 8,
            Ty::I64 => 8,
            Ty::USize => 8,
            Ty::Ptr {..} => 8,
            Ty::Array { elem_type, .. } => elem_type.get_align(),
            Ty::Record { align, .. } => *align,
            Ty::Var(_) | Ty::Void | Ty::Func {..}
                => unreachable!(),
        }
    }

    pub fn get_size(&self) -> usize {
        match self {
            Ty::Bool | Ty::U8 | Ty::I8 => 1,
            Ty::U16 => 2,
            Ty::I16 => 2,
            Ty::U32 => 4,
            Ty::I32 => 4,
            Ty::U64 => 8,
            Ty::I64 => 8,
            Ty::USize => 8,
            Ty::Ptr {..} => 8,
            Ty::Array { elem_type, elem_count }
                => elem_type.get_size() * elem_count
                        .expect("Array without element count allocated"),
            Ty::Record { size, .. } => *size,
            Ty::Var(_) | Ty::Void | Ty::Func {..}
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
    Global(Rc<str>),
    Local(Rc<Local>),
    // Postfix expressions
    Field(Box<Expr>, usize),
    Call(Box<Expr>, Vec<Expr>, bool),
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
    pub ty: Ty,
    pub kind: ExprKind,
}

impl Expr {
    fn make_const(ty: Ty, val: usize) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Const(val)
        }
    }

    fn make_global(ty: Ty, name: Rc<str>) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Global(name),
        }
    }

    fn make_local(ty: Ty, local: Rc<Local>) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Local(local),
        }
    }
}

#[derive(Debug)]
pub enum Stmt {
    Block(Vec<Stmt>),
    Eval(Expr),
    Ret(Option<Expr>),
    Auto(Ty, Rc<Local>, Option<Expr>),
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
    list: Vec<HashMap<Rc<str>, Expr>>,
}

impl SymTab {
    fn new() -> SymTab {
        let mut symtab = SymTab {
            list: Vec::new(),
        };
        symtab.list.push(HashMap::new());
        symtab
    }

    fn insert(&mut self, name: Rc<str>, expr: Expr) {
        let scope = self.list.last_mut().unwrap();
        if let None = scope.get(&name) {
            scope.insert(name, expr);
        } else {
            panic!("Re-declaration of {}", name)
        }
    }

    fn lookup(&mut self, name: &Rc<str>) -> &Expr {
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

struct Parser<'a> {
    ts: PeekIter<Lexer<'a>, 2>,
    // Code generation backend
    gen: &'a mut Gen,
    // Currently defined record types
    records: HashMap<Rc<str>, Ty>,
    // Symbol table
    symtab: SymTab,
    // Ty variable index
    tvar: usize,
    // Ty variable constraints
    tvarmap: HashMap<usize, Ty>,
}

impl<'a> Parser<'a> {
    fn new(lexer: Lexer<'a>, gen: &'a mut Gen) -> Self {
        Parser {
            ts: PeekIter::new(lexer),
            gen: gen,
            records: HashMap::new(),
            symtab: SymTab::new(),
            tvar: 0,
            tvarmap: HashMap::new(),
        }
    }

    // Read an identifier
    fn next_ident(&mut self) -> Rc<str> {
        match self.ts.next() {
            Some(Token::Ident(val)) => val,
            got => panic!("Expected identifier, got {:?}", got),
        }
    }

    // Create a new unique type variable
    fn next_tvar(&mut self) -> Ty {
        let tvar = self.tvar;
        self.tvar += 1;
        Ty::Var(tvar)
    }

    // Unify two types
    fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> Ty {
        match (ty1, ty2) {
            // Type variables
            (Ty::Var(var), newty) | (newty, Ty::Var(var)) => {
                let ty = if let Some(prevty) = self.tvarmap.remove(&var) {
                    // If there was a previous bound for this variable,
                    // unify them and insert the unified type
                    self.unify(&prevty, &newty)
                } else {
                    // If not, insert the new type into the table
                    newty.clone()
                };
                self.tvarmap.insert(*var, ty.clone());
                ty
            },

            // Aggregate types are composed of other types
            (Ty::Ptr { base_type: base_ty1 }, Ty::Ptr { base_type: base_ty2 })
                => Ty::Ptr {
                    base_type: Box::new(self.unify(base_ty1, base_ty2))
                },

            (Ty::Array { elem_type: elem_ty1, elem_count: cnt1 },
             Ty::Array { elem_type: elem_ty2, elem_count: cnt2 })
                => Ty::Array {
                    elem_type: Box::new(self.unify(elem_ty1, elem_ty2)),
                    elem_count:
                        // Unknown length arrays come from array types created
                        // when indexing
                        match (*cnt1, *cnt2) {
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

            (Ty::Record { name: name1, .. }, Ty::Record { name: name2, .. })
                => if name1 != name2 {
                    panic!("Incompatible types {:?}, {:?}", ty1, ty2);
                } else {
                    ty1.clone()
                }

            (Ty::Func { params: params1, varargs: varargs1, rettype: rettype1 },
             Ty::Func { params: params2, varargs: varargs2, rettype: rettype2 })
                => {
                    // Make sure the varargs designators match
                    if varargs1 != varargs2 {
                        panic!("Tried to unify varargs function with a non-varargs one");
                    }

                    // Unify parameters and return type
                    Ty::Func {
                        params: params1.iter()
                                        .zip(params2.iter())
                                        .map(|(p1,p2)| self.unify(p1, p2))
                                        .collect(),
                        varargs: *varargs1,
                        rettype: Box::new(self.unify(rettype1, rettype2))
                    }
                }

            // Base case: basic types
            (Ty::U8, Ty::U8) => Ty::U8,
            (Ty::I8, Ty::I8) => Ty::I8,
            (Ty::U16, Ty::U16) => Ty::U16,
            (Ty::I16, Ty::I16) => Ty::I16,
            (Ty::U32, Ty::U32) => Ty::U32,
            (Ty::I32, Ty::I32) => Ty::I32,
            (Ty::U64, Ty::U64) => Ty::U64,
            (Ty::I64, Ty::I64) => Ty::I64,
            (Ty::USize, Ty::USize) => Ty::USize,
            (Ty::Bool, Ty::Bool) => Ty::Bool,
            (Ty::Void, Ty::Void) => Ty::Void,

            // Base case: incompatible types
            (ty1, ty2)
                => panic!("Incompatible types {:?}, {:?}", ty1, ty2),
        }
    }

    // Find the literal type for a type expression that might contain type variables
    // NOTE: only works after all constraints were unified
    fn lit_type(&mut self, ty: Ty) -> Ty {
        match ty {
            // Recurse: aggregate types are composed of other types
            // NOTE: records and functions aren't treated as aggregates for now
            // as those form boundaries for type deduction and must be explictly
            // annotated
            Ty::Ptr { base_type }
                => Ty::Ptr {
                    base_type: Box::new(self.lit_type(*base_type))
                },
            Ty::Array { elem_type, elem_count }
                => Ty::Array {
                    elem_type: Box::new(self.lit_type(*elem_type)),
                    elem_count: elem_count
                },

            // Base case: type variables
            Ty::Var(var)
                => if let Some(ty) = self.tvarmap.remove(&var) {
                    let ty = self.lit_type(ty);
                    self.tvarmap.insert(var, ty.clone());
                    ty
                } else {
                    panic!("Ty variable {} not deducible, provide more information", var);
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

    fn make_field(&mut self, record: Expr, name: Rc<str>) -> Expr {
        // Records are type deduction boundaries,
        // we can finish deducing here and enforce.
        if let Ty::Record { lookup, fields, .. } = self.lit_type(record.ty.clone()) {
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
        if let Ty::Func { params, varargs, rettype } = self.lit_type(func.ty.clone()) {
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
                self.unify(&param_ty, &arg.ty);
            }

            Expr {
                ty: *rettype.clone(),
                kind: ExprKind::Call(Box::new(func), args, varargs)
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
        self.unify(&array.ty, &Ty::Array {
            elem_type: Box::new(elem_ty.clone()),
            elem_count: None,
        });
        self.unify(&index.ty, &Ty::USize);
        Expr {
            ty: elem_ty,
            kind: ExprKind::Elem(Box::new(array), Box::new(index))
        }
    }


    fn make_ref(&mut self, expr: Expr) -> Expr {
        Expr {
            ty: Ty::Ptr { base_type: Box::new(expr.ty.clone()) },
            kind: ExprKind::Ref(Box::new(expr))
        }
    }

    fn make_deref(&mut self, ptr: Expr) -> Expr {
        // We are doing the same thing here as in make_elem() above
        let base_ty = self.next_tvar();
        self.unify(&ptr.ty, &Ty::Ptr {
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

    fn make_cast(&mut self, expr: Expr, ty: Ty) -> Expr {
        Expr {
            ty: ty,
            kind: ExprKind::Cast(Box::new(expr))
        }
    }

    fn make_mul(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(&expr.ty, &expr2.ty),
            kind: ExprKind::Mul(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_div(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(&expr.ty, &expr2.ty),
            kind: ExprKind::Div(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_rem(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(&expr.ty, &expr2.ty),
            kind: ExprKind::Rem(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_add(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(&expr.ty, &expr2.ty),
            kind: ExprKind::Add(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_sub(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(&expr.ty, &expr2.ty),
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
        self.unify(&expr.ty, &expr2.ty);
        Expr {
            ty: Ty::Bool,
            kind: ExprKind::Lt(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_le(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(&expr.ty, &expr2.ty);
        Expr {
            ty: Ty::Bool,
            kind: ExprKind::Le(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_gt(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(&expr.ty, &expr2.ty);
        Expr {
            ty: Ty::Bool,
            kind: ExprKind::Gt(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_ge(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(&expr.ty, &expr2.ty);
        Expr {
            ty: Ty::Bool,
            kind: ExprKind::Ge(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_eq(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(&expr.ty, &expr2.ty);
        Expr {
            ty: Ty::Bool,
            kind: ExprKind::Eq(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_ne(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(&expr.ty, &expr2.ty);
        Expr {
            ty: Ty::Bool,
            kind: ExprKind::Ne(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_and(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(&expr.ty, &expr2.ty),
            kind: ExprKind::And(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_xor(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(&expr.ty, &expr2.ty),
            kind: ExprKind::Xor(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_or(&mut self, expr: Expr, expr2: Expr) -> Expr {
        Expr {
            ty: self.unify(&expr.ty, &expr2.ty),
            kind: ExprKind::Or(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_land(&mut self, expr: Expr, expr2: Expr) -> Expr {
        // Mostly pointless, but we can add a constraint on these being bool
        // to the deducer. This should not matter for valid code, but for example
        // this will force un-initialized lets used in this context to be deduced
        // to be bool.
        self.unify(&expr.ty, &Ty::Bool);
        self.unify(&expr2.ty, &Ty::Bool);
        Expr {
            ty: Ty::Bool,
            kind: ExprKind::LAnd(Box::new(expr), Box::new(expr2))
        }
    }

    fn make_lor(&mut self, expr: Expr, expr2: Expr) -> Expr {
        self.unify(&expr.ty, &Ty::Bool);
        self.unify(&expr2.ty, &Ty::Bool);
        Expr {
            ty: Ty::Bool,
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
            kind @ (ExprKind::Const(_) | ExprKind::Global(_) | ExprKind::Local(_)) => kind,
            ExprKind::Compound(exprs)
                => ExprKind::Compound(exprs.into_iter()
                    .map(|expr| self.finalize_expr(expr)).collect()),

            ExprKind::Field(record, off)
                => ExprKind::Field(Box::new(self.finalize_expr(*record)), off),
            ExprKind::Call(func, args, varargs)
                => ExprKind::Call(Box::new(self.finalize_expr(*func)),
                    args.into_iter().map(|expr| self.finalize_expr(expr)).collect(),
                    varargs),
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
            Stmt::Auto(mut ty, local, mut opt_expr) => {
                // Obtain literal type for declaration
                ty = self.lit_type(ty);

                // Finalize initializer if present
                if let Some(expr) = opt_expr {
                    opt_expr = Some(self.finalize_expr(expr));
                }
                Stmt::Auto(ty, local, opt_expr)
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

    // Read a visibility (or return Vis::Private)
    fn maybe_want_vis(&mut self) -> Vis {
        if maybe_want!(self.ts, Token::Export) {
            Vis::Export
        } else if maybe_want!(self.ts, Token::Extern) {
            Vis::Extern
        } else {
            Vis::Private
        }
    }

    // Read an integer type (or return None)
    fn want_type_suffix(&mut self) -> Option<Ty> {
        // Match for type suffix
        let suf = match self.ts.peek(0) {
            Some(Token::U8)   => Ty::U8,
            Some(Token::I8)   => Ty::I8,
            Some(Token::U16)  => Ty::U16,
            Some(Token::I16)  => Ty::I16,
            Some(Token::U32)  => Ty::U32,
            Some(Token::I32)  => Ty::I32,
            Some(Token::U64)  => Ty::U64,
            Some(Token::I64)  => Ty::I64,
            Some(Token::USize)=> Ty::USize,
            _ => return None,
        };
        // Remove matched token from stream
        self.ts.next();
        Some(suf)
    }

    fn want_array_literal(&mut self) -> Expr {
        let mut elem_ty = self.next_tvar();
        let mut elems = Vec::new();

        while !maybe_want!(self.ts, Token::RSq) {
            let expr = self.want_expr();
            elem_ty = self.unify(&elem_ty, &expr.ty);
            elems.push(expr);
            if !maybe_want!(self.ts, Token::Comma) {
                want!(self.ts, Token::RSq);
                break;
            }
        }

        Expr {
            ty: Ty::Array {
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

        let (lookup, fields) = if let Ty::Record { lookup, fields, .. } = &ty {
            (lookup, fields)
        } else {
            unreachable!()
        };

        // Read expressions
        let mut expr_map = HashMap::new();
        while !maybe_want!(self.ts, Token::RCurly) {
            // Read field name
            let name = self.next_ident();

            // Check for duplicate field
            if let Some(_) = expr_map.get(&name) {
                panic!("Duplicate field {}", name);
            }

            // Read field initializer
            want!(self.ts, Token::Colon);
            expr_map.insert(name, self.want_expr());

            // See if we've reached the end
            if !maybe_want!(self.ts, Token::Comma) {
                want!(self.ts, Token::RCurly);
                break;
            }
        }

        // Map expressions to fields
        let mut expr_vec = Vec::new();
        for (name, idx) in lookup {
            let ty = fields[*idx].0.clone();
            if let Some(expr) = expr_map.remove(name) {
                self.unify(&ty, &expr.ty);
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
        match self.ts.next().expect("Expected primary expression") {
            Token::LParen => {
                let expr = self.want_expr();
                want!(self.ts, Token::RParen);
                expr
            },
            Token::Str(data) => {
                let chty = self.want_type_suffix().unwrap_or(Ty::U8);
                let (name, ty) = self.gen.do_string(chty.clone(), &*data);
                // Replace string literal with reference to internal symbol
                let sym_expr = Expr::make_global(ty, name);
                let mut ref_expr = self.make_ref(sym_expr);
                // HACK: this reference doesn't actually have an array pointer's
                // type, because C APIs degrade arrays to pointers to the first
                // element
                ref_expr.ty = Ty::Ptr { base_type: Box::new(chty) };
                ref_expr
            },
            Token::Ident(name)
                => if maybe_want!(self.ts, Token::LCurly) {
                    self.want_record_literal(name)
                } else {
                    self.symtab.lookup(&name).clone()
                },
            Token::LSq
                => self.want_array_literal(),
            Token::Constant(val) => {
                let ty = if let Some(ty) = self.want_type_suffix() {
                    ty
                } else {
                    self.next_tvar()
                };
                Expr::make_const(ty, val)
            },
            Token::True => Expr::make_const(Ty::Bool, 1),
            Token::False => Expr::make_const(Ty::Bool, 0),

            tok => panic!("Invalid primary expression {:?}", tok),
        }
    }

    fn want_postfix(&mut self) -> Expr {
        let mut expr = self.want_primary();
        loop {
            if maybe_want!(self.ts, Token::Dot) {
                let name = self.next_ident();
                expr = self.make_field(expr, name);
            } else if maybe_want!(self.ts, Token::LParen) {
                let mut args = Vec::new();
                while !maybe_want!(self.ts, Token::RParen) {
                    args.push(self.want_expr());
                    if !maybe_want!(self.ts, Token::Comma) {
                        want!(self.ts, Token::RParen);
                        break;
                    }
                }
                expr = self.make_call(expr, args);
            } else if maybe_want!(self.ts, Token::LSq) {
                let index = self.want_expr();
                expr = self.make_elem(expr, index);
                want!(self.ts, Token::RSq);
            } else {
                return expr;
            }
        }
    }

    fn want_unary(&mut self) -> Expr {
        if maybe_want!(self.ts, Token::Sub) {
            let expr = self.want_unary();
            self.make_neg(expr)
        } else if maybe_want!(self.ts, Token::Tilde) {
            let expr = self.want_unary();
            self.make_not(expr)
        } else if maybe_want!(self.ts, Token::Excl) {
            let expr = self.want_unary();
            self.make_lnot(expr)
        } else if maybe_want!(self.ts, Token::Mul) {
            let expr = self.want_unary();
            self.make_deref(expr)
        } else if maybe_want!(self.ts, Token::And) {
            let expr = self.want_unary();
            self.make_ref(expr)
        } else if maybe_want!(self.ts, Token::Add) {
            self.want_unary()
        } else {
            self.want_postfix()
        }
    }

    fn want_cast(&mut self) -> Expr {
        let mut expr = self.want_unary();
        while maybe_want!(self.ts, Token::As) {
            let ty = self.want_type();
            expr = self.make_cast(expr, ty);
        }
        expr
    }

    fn want_mul(&mut self) -> Expr {
        let mut expr = self.want_cast();
        loop {
            if maybe_want!(self.ts, Token::Mul) {
                let expr2 = self.want_cast();
                expr = self.make_mul(expr, expr2);
            } else if maybe_want!(self.ts, Token::Div) {
                let expr2 = self.want_cast();
                expr = self.make_div(expr, expr2);
            } else if maybe_want!(self.ts, Token::Rem) {
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
            if maybe_want!(self.ts, Token::Add) {
                let expr2 = self.want_mul();
                expr = self.make_add(expr, expr2);
            } else if maybe_want!(self.ts, Token::Sub) {
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
            if maybe_want!(self.ts, Token::Lsh) {
                let expr2 = self.want_add();
                expr = self.make_lsh(expr, expr2);
            } else if maybe_want!(self.ts, Token::Rsh) {
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
            if maybe_want!(self.ts, Token::And) {
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
            if maybe_want!(self.ts, Token::Xor) {
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
            if maybe_want!(self.ts, Token::Or) {
                let expr2 = self.want_xor();
                expr = self.make_or(expr, expr2);
            } else {
                return expr;
            }
        }
    }

    fn want_cmp(&mut self) -> Expr {
        let expr = self.want_or();
        if maybe_want!(self.ts, Token::Lt) {
            let expr2 = self.want_or();
            self.make_lt(expr, expr2)
        } else if maybe_want!(self.ts, Token::Le) {
            let expr2 = self.want_or();
            self.make_le(expr, expr2)
        } else if maybe_want!(self.ts, Token::Gt) {
            let expr2 = self.want_or();
            self.make_gt(expr, expr2)
        } else if maybe_want!(self.ts, Token::Ge) {
            let expr2 = self.want_or();
            self.make_ge(expr, expr2)
        } else if maybe_want!(self.ts, Token::Eq) {
            let expr2 = self.want_or();
            self.make_eq(expr, expr2)
        } else if maybe_want!(self.ts, Token::Ne) {
            let expr2 = self.want_or();
            self.make_ne(expr, expr2)
        } else {
            expr
        }
    }

    fn want_land(&mut self) -> Expr {
        let mut expr = self.want_cmp();
        loop {
            if maybe_want!(self.ts, Token::LAnd) {
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
            if maybe_want!(self.ts, Token::LOr) {
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

    fn want_type(&mut self) -> Ty {
        match self.ts.next().expect("Expected type") {
            Token::Bool => Ty::Bool,
            Token::U8   => Ty::U8,
            Token::I8   => Ty::I8,
            Token::U16  => Ty::U16,
            Token::I16  => Ty::I16,
            Token::U32  => Ty::U32,
            Token::I32  => Ty::I32,
            Token::U64  => Ty::U64,
            Token::I64  => Ty::I64,
            Token::USize=> Ty::USize,
            Token::Mul  => Ty::Ptr {
                base_type: Box::new(self.want_type())
            },
            Token::LSq  => {
                let elem_type = self.want_type();
                want!(self.ts, Token::Semicolon);
                let expr = self.want_expr();
                self.unify(&expr.ty, &Ty::USize);
                let elem_count = match self.finalize_expr(expr).kind {
                    ExprKind::Const(val) => val,
                    _ => panic!("Array element count must be constnat"),
                };
                want!(self.ts, Token::RSq);

                Ty::Array {
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
                want!(self.ts, Token::LParen);
                let mut params = Vec::new();
                let mut varargs = false;
                while !maybe_want!(self.ts, Token::RParen) {
                    if maybe_want!(self.ts, Token::Varargs) {
                        varargs = true;
                        want!(self.ts, Token::LParen);
                        break;
                    }
                    params.push(self.want_type());
                    if !maybe_want!(self.ts, Token::Comma) {
                        want!(self.ts, Token::RParen);
                        break;
                    }
                }

                // Read return type
                let rettype = if maybe_want!(self.ts, Token::Arrow) {
                    self.want_type()
                } else {
                    Ty::Void
                };

                Ty::Func {
                    params: params.into(),
                    varargs: varargs,
                    rettype: Box::new(rettype)
                }
            },
            _ => panic!("Invalid typename!"),
        }
    }

    fn want_record(&mut self, name: Rc<str>) -> Ty {
        fn round_up(val: usize, bound: usize) -> usize {
            (val + bound - 1) / bound * bound
        }

        let mut lookup = HashMap::new();
        let mut fields = Vec::new();
        let mut max_align = 0;
        let mut offset = 0;

        want!(self.ts, Token::LCurly);

        // Read fields until }
        while !maybe_want!(self.ts, Token::RCurly) {
            lookup.insert(self.next_ident(), fields.len());
            want!(self.ts, Token::Colon);

            let field_type = self.want_type();
            let field_align = field_type.get_align();
            let field_size = field_type.get_size();

            // Save maximum alignment of all fields
            if max_align < field_align {
                max_align = field_align;
            }

            // Round field offset to correct alignment
            offset = round_up(offset, field_align);
            fields.push((field_type, offset));
            offset += field_size;

            if !maybe_want!(self.ts, Token::Comma) {
                want!(self.ts, Token::RCurly);
                break;
            }
        }

        // Record type declaration must end in semicolon
        want!(self.ts, Token::Semicolon);

        Ty::Record {
            name: name,
            lookup: lookup,
            fields: fields.into_boxed_slice(),
            align: max_align,
            // Round struct size to a multiple of it's alignment, this is needed
            // if an array is ever created of this struct
            size: round_up(offset, max_align),
        }
    }

    fn want_stmts(&mut self, rettype: &Ty) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        while !maybe_want!(self.ts, Token::RCurly) {
            stmts.push(self.want_stmt(rettype));
        }
        stmts
    }

    fn want_block(&mut self, rettype: &Ty) -> Stmt {
        self.symtab.push_scope();
        let stmts = self.want_stmts(rettype);
        self.symtab.pop_scope();
        Stmt::Block(stmts)
    }

    fn want_if(&mut self, rettype: &Ty) -> Stmt {
        // Read conditional
        want!(self.ts, Token::LParen);
        let cond = self.want_expr();
        want!(self.ts, Token::RParen);

        // Read true branch
        want!(self.ts, Token::LCurly);
        let then = self.want_block(rettype);

        // Read else branch if present
        let _else = if maybe_want!(self.ts, Token::Else) {
            let else_body = match self.ts.next() {
                Some(Token::LCurly) => self.want_block(rettype),
                Some(Token::If)     => self.want_if(rettype),
                _                   => panic!("Expected block or if after else")
            };
            Some(Box::new(else_body))
        } else {
            None
        };

        Stmt::If(cond, Box::new(then), _else)
    }

    fn want_while(&mut self, rettype: &Ty) -> Stmt {
        // Read conditional
        want!(self.ts, Token::LParen);
        let cond = self.want_expr();
        want!(self.ts, Token::RParen);

        // Read body
        want!(self.ts, Token::LCurly);
        let body = self.want_block(rettype);

        Stmt::While(cond, Box::new(body))
    }

    fn want_eval_or_set(&mut self) -> Stmt {
        let expr = self.want_expr();
        let stmt = if maybe_want!(self.ts, Token::Assign) {
            let src = self.want_expr();
            self.unify(&expr.ty, &src.ty);
            Stmt::Set(expr, src)
        } else {
            Stmt::Eval(expr)
        };
        want!(self.ts, Token::Semicolon);
        stmt
    }

    fn want_stmt(&mut self, rettype: &Ty) -> Stmt {
        match self.ts.peek(0).expect("Expected statement") {
            // Nested scope
            Token::LCurly => {
                self.ts.next(); // Skip {
                self.want_block(rettype)
            },

            // Conditional
            Token::If => {
                self.ts.next(); // Skip if
                self.want_if(rettype)
            },

            // While loop
            Token::While => {
                self.ts.next(); // Skip while
                self.want_while(rettype)
            },

            // Statements
            Token::Let => {
                self.ts.next(); // Skip let

                // Read declared name
                let name = self.next_ident();
                // Read type (or create type variable)
                let mut ty = if maybe_want!(self.ts, Token::Colon) {
                    self.want_type()
                } else {
                    self.next_tvar()
                };

                // Unify type with initializer type
                let opt_expr = if maybe_want!(self.ts, Token::Assign) {
                    let expr = self.want_expr();
                    ty = self.unify(&ty, &expr.ty);
                    Some(expr)
                } else {
                    None
                };

                // Insert symbol
                let local = Rc::new(Local::new());
                self.symtab.insert(name, Expr::make_local(ty.clone(), local.clone()));

                // Create statement
                let stmt = Stmt::Auto(ty, local, opt_expr);
                want!(self.ts, Token::Semicolon);
                stmt
            },

            Token::Ident(_) => {
                // This could either be a label or a start of an expression,
                // we need two tokens lookahead to tell these productions apart
                if let Some(Token::Colon) = self.ts.peek(1) {
                    let stmt = Stmt::Label(self.next_ident());   // Read label
                    self.ts.next();                                 // Skip :
                    stmt
                } else {
                    self.want_eval_or_set()
                }
            },

            Token::Jmp => {
                self.ts.next(); // Skip jmp
                let stmt = Stmt::Jmp(self.next_ident());
                want!(self.ts, Token::Semicolon);
                stmt
            },

            Token::Ret => {
                self.ts.next(); // Skip ret
                if maybe_want!(self.ts, Token::Semicolon) {
                    Stmt::Ret(None)
                } else {
                    let expr = self.want_expr();
                    self.unify(&expr.ty, rettype);
                    let stmt = Stmt::Ret(Some(expr));
                    want!(self.ts, Token::Semicolon);
                    stmt
                }
            },

            _ => self.want_eval_or_set(),
        }
    }

    fn process(&mut self) {
        while let Some(token) = self.ts.next() {
            //
            // Process file element
            //

            match token {
                Token::Record => {
                    let name = self.next_ident();
                    let record = self.want_record(name.clone());
                    self.records.insert(name, record);
                },
                Token::Static => {
                    let vis = self.maybe_want_vis();
                    let name = self.next_ident();
                    // Link symbol as requested
                    self.gen.do_link(name.clone(), vis);

                    let mut ty = if maybe_want!(self.ts, Token::Colon) {
                        self.want_type()
                    } else {
                        self.next_tvar()
                    };

                    if maybe_want!(self.ts, Token::Assign) {
                        // Unify initializer type with symbol type
                        let mut expr = self.want_expr();
                        ty = self.unify(&ty, &expr.ty);
                        expr = self.finalize_expr(expr);

                        // Create data
                        self.gen.do_data(&name, &expr);
                    } else {
                        // Reserve bss space
                        self.gen.do_bss(&name, &ty);
                    }

                    // Add symbol
                    self.symtab.insert(name.clone(), Expr::make_global(ty, name));

                    want!(self.ts, Token::Semicolon);
                },
                Token::Fn => {
                    let vis = self.maybe_want_vis();
                    let name = self.next_ident();
                    // Link symbol as requested
                    self.gen.do_link(name.clone(), vis);

                    let mut params = Vec::new();
                    let mut varargs = false;

                    let mut param_name_ty = Vec::new();

                    // Read parameters
                    want!(self.ts, Token::LParen);
                    while !maybe_want!(self.ts, Token::RParen) {
                        // Last parameter can be varargs
                        if maybe_want!(self.ts, Token::Varargs) {
                            varargs = true;
                            want!(self.ts, Token::RParen);
                            break;
                        }
                        // Otherwise try reading a normal parameter
                        let name = self.next_ident();
                        want!(self.ts, Token::Colon);
                        let ty = self.want_type();
                        params.push(ty.clone());
                        param_name_ty.push((name, ty));
                        if !maybe_want!(self.ts, Token::Comma) {
                            want!(self.ts, Token::RParen);
                            break;
                        }
                    }

                    // Read return type (or set to void)
                    let rettype = if maybe_want!(self.ts, Token::Arrow) {
                        self.want_type()
                    } else {
                        Ty::Void
                    };

                    // Create symbol for function
                    let ty = Ty::Func {
                        params: params.into(),
                        varargs: varargs,
                        rettype: Box::new(rettype.clone()),
                    };
                    self.symtab.insert(name.clone(), Expr::make_global(ty, name.clone()));

                    // Read body (if present)
                    if maybe_want!(self.ts, Token::LCurly) {
                        self.symtab.push_scope();

                        // Add paramters to scope
                        let mut params = Vec::new();
                        for (name, ty) in param_name_ty.into_iter() {
                            let local = Rc::new(Local::new());
                            params.push((ty.clone(), local.clone()));
                            self.symtab.insert(name, Expr::make_local(ty, local));
                        }

                        // Read body
                        let mut stmts = self.want_stmts(&rettype);

                        self.symtab.pop_scope();

                        // Generate body
                        // println!("AST: {:#?}", stmts);
                        // println!("Constraints: {:#?}", self.tvarmap);

                        stmts = stmts.into_iter()
                            .map(|stmt| self.finalize_stmt(stmt)).collect();

                        // println!("Deduced AST: {:#?}", stmts);

                        self.gen.do_func(name, params, stmts);
                    } else {
                        want!(self.ts, Token::Semicolon)
                    }
                },
                _ => panic!("Expected record, union, static or function!"),
            }

            //
            // Clear file element specific data
            //

            self.tvar = 0;
            self.tvarmap.clear();
        }
    }
}

pub fn parse_file(data: &str, gen: &mut Gen) {
    let mut parser = Parser::new(Lexer::new(data), gen);
    parser.process();
}
