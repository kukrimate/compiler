// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation interface
//

// pub mod nasm;
pub mod s16;

use crate::ast::{Ty,Vis};
use std::rc::Rc;

pub trait DataGen {
    // Generate an immediate constant
    fn _const(&mut self, ty: &Ty, val: usize);
    // Generate the address of a global (to be resolved at link time)
    fn addr(&mut self, name: &Rc<str>);
}

#[derive(Clone,Copy,PartialEq)]
pub enum UOp {
    Not,
    Neg,
}

#[derive(Clone,Copy,PartialEq)]
pub enum BOp {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Lsh,
    Rsh,
    And,
    Xor,
    Or,
}

#[derive(Clone,Copy)]
pub enum Cond {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

pub trait FuncGen {
    type Val: Clone;
    type Label: Clone + Copy;

    // Allocate a local variable
    fn next_local(&mut self, ty: &Ty) -> Self::Val;
    // Allocate a new unique label
    fn next_label(&mut self) -> Self::Label;

    // Generate a jump target for "label"
    fn tgt(&mut self, label: Self::Label);
    // Generate a jump to "label"
    fn jmp(&mut self, label: Self::Label);

    // Move the next parameter to a location
    fn arg_to_val(&mut self, ty: &Ty, dst: Self::Val);
    // Move a value to the return location
    fn val_to_ret(&mut self, ty: &Ty, src: Self::Val);

    // Convert a bool to a value
    fn bool_to_val(&mut self, ltrue: Self::Label, lfalse: Self::Label) -> Self::Val;
    // Convert a value to a bool
    fn val_to_bool(&mut self, val: Self::Val, ltrue: Self::Label, lfalse: Self::Label);

    // Generate an immediate value
    fn immed(&mut self, val: usize) -> Self::Val;
    // Generate a global variable
    fn global(&mut self, name: Rc<str>) -> Self::Val;
    // Allocate a compound literal
    fn compound(&mut self, ty: &Ty, vals: Vec<(&Ty, Self::Val)>) -> Self::Val;

    // Access a struct field "off" bytes from val
    fn field(&mut self, val: Self::Val, off: usize) -> Self::Val;
    // Access the array element at "index" in an array with "stride" byte elements
    fn elem(&mut self, array: Self::Val, index: Self::Val, stride: usize) -> Self::Val;
    // Call the function at "func"
    fn call(&mut self, retty: &Ty, varargs: bool,
            func: Self::Val,
            args: Vec<(&Ty, Self::Val)>) -> Self::Val;

    // Create a pointer to val
    fn _ref(&mut self, val: Self::Val) -> Self::Val;
    // Dereference the pointer at val
    fn deref(&mut self, val: Self::Val) -> Self::Val;

    // Perform the op (unary) operation on val
    fn unary(&mut self, op: UOp, ty: &Ty, val: Self::Val) -> Self::Val;

    // Perform the op (binary) operation on val
    fn binary(&mut self, op: BOp, ty: &Ty, v1: Self::Val, v2: Self::Val) -> Self::Val;

    // Generate a conditional branch to "ltrue" or "lfalse" based on a comparison
    fn jcc(&mut self, cond: Cond, ltrue: Self::Label, lfalse: Self::Label,
            ty: &Ty, v1: Self::Val, v2: Self::Val);

    // Generate an assignment
    fn assign(&mut self, ty: &Ty, dst: Self::Val, src: Self::Val);
}

pub trait Gen {
    type DataGen: DataGen;
    type FuncGen: FuncGen;

    // Add a linkage marker for an identifier
    fn do_link(&mut self, name: Rc<str>, vis: Vis);

    // Create a string literal in the backend
    fn do_string(&mut self, chty: Ty, data: &str) -> (Rc<str>, Ty);

    // Perform a bss allocation as "name" for "ty".get_size() bytes
    fn do_bss(&mut self, name: &Rc<str>, ty: &Ty);

    // Merge-in some generated data
    fn begin_data(&mut self, name: &Rc<str>) -> Self::DataGen;
    fn end_data(&mut self, data_gen: Self::DataGen);

    // Merge-in a generated function
    fn begin_func(&mut self, name: &Rc<str>) -> Self::FuncGen;
    fn end_func(&mut self, func_gen: Self::FuncGen);
}

