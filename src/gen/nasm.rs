// SPDX-License-Identifier: GPL-2.0-only

//
// Code generation backend for nasm
//

use crate::ast::{Ty,Vis};
use super::{DataGen,UOp,BOp,Cond,FuncGen,Gen};

use std::fmt::{Display,Formatter,Write};
use std::rc::Rc;

//
// Operation widths supported by x86
//

#[derive(Clone,Copy)]
enum Width {
    Byte    = 1,
    Word    = 2,
    DWord   = 4,
    QWord   = 8,
}

fn loc_str(width: Width) -> &'static str {
    match width {
        Width::Byte => "byte",
        Width::Word => "word",
        Width::DWord => "dword",
        Width::QWord => "qword",
    }
}

fn data_str(width: Width) -> &'static str {
    match width {
        Width::Byte => "db",
        Width::Word => "dw",
        Width::DWord => "dd",
        Width::QWord => "dq",
    }
}

// Find the largest width operation possible on size bytes
fn max_width(size: usize) -> Width {
    for width in [ Width::QWord, Width::DWord, Width::Word, Width::Byte ] {
        if width as usize <= size {
            return width;
        }
    }
    unreachable!();
}

// Find the register size used for a parameter
fn type_width(ty: &Ty) -> Width {
    match ty {
        Ty::Bool | Ty::U8 | Ty::I8 => Width::Byte,
        Ty::U16 | Ty::I16 => Width::Word,
        Ty::U32 | Ty::I32 => Width::DWord,
        Ty::U64 | Ty::I64 | Ty::USize | Ty::Ptr{..} => Width::QWord,
        _ => unreachable!(),
    }
}

#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
enum Reg {
    Rax = 0,
    Rbx = 1,
    Rcx = 2,
    Rdx = 3,
    Rsi = 4,
    Rdi = 5,
    R8  = 6,
    R9  = 7,
    /*
    R10 = 8,
    R11 = 9,
    R12 = 10,
    R13 = 11,
    R14 = 12,
    R15 = 13,
    */
}


//
// Register parameter order
//

const PARAMS: [Reg; 6] = [ Reg::Rdi, Reg::Rsi, Reg::Rdx, Reg::Rcx, Reg::R8, Reg::R9 ];

//
// All usable registers (in preferred allocation order)
//

/*
const ALL_REGS: [Reg; 14] = [
    Reg::Rbx, Reg::R12, Reg::R13, Reg::R14, Reg::R15,           // Callee saved (using these is free)
    Reg::Rsi, Reg::Rdi, Reg::R8, Reg::R9, Reg::R10, Reg::R11,   // Never needed
    Reg::Rcx, Reg::Rdx, Reg::Rax, ];                            // Sometimes needed
*/

fn reg_str(width: Width, reg: Reg) -> &'static str {
    match width {
        Width::Byte
            => ["al", "bl", "cl", "dl", "sil", "dil", "r8b", "r9b", "r10b",
                "r11b", "r12b", "r13b", "r14b", "r15b"][reg as usize],
        Width::Word
            => ["ax", "bx", "cx", "dx", "si", "di", "r8w", "r9w", "r10w",
                "r11w", "r12w", "r13w", "r14w", "r15w"][reg as usize],
        Width::DWord
            => ["eax", "ebx", "ecx", "edx", "esi", "edi", "r8d", "r9d", "r10d",
                "r11d", "r12d", "r13d", "r14d", "r15d"][reg as usize],
        Width::QWord
            => ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10",
                "r11", "r12", "r13", "r14", "r15"][reg as usize],
    }
}

//
// Unique labels
//

#[derive(Clone,Copy)]
pub struct Label(usize);

impl Display for Label {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(fmt, "{}", self.0)
    }
}

//
// Promise for a runtime value
//

#[derive(Clone,Debug,Hash,PartialEq,Eq)]
pub enum Val {
    Void,                     // Non-existent value
    Imm(usize),               // Immediate constant
    Loc(usize),               // Reference to stack
    Glo(Rc<str>, usize),      // Reference to symbol
    Deref(Box<Val>, usize),   // De-reference of pointer
}

impl Val {
    fn with_offset(self, add: usize) -> Val {
        match self {
            Val::Void | Val::Imm(_) => unreachable!(),
            Val::Loc(offset) => Val::Loc(offset + add),
            Val::Glo(name, offset) => Val::Glo(name, offset + add),
            Val::Deref(ptr, offset) => Val::Deref(ptr, offset + add),
        }
    }
}

//
// Code generator implementation
//

pub struct DataGenNasm {
    data: String
}

impl DataGen for DataGenNasm {
    fn _const(&mut self, ty: &Ty, val: usize) {
        writeln!(self.data, "{} {}", data_str(type_width(ty)), val).unwrap();
    }

    fn addr(&mut self, name: &Rc<str>) {
        writeln!(self.data, "dq {}", name).unwrap();
    }
}

pub struct FuncGenNasm {
    frame_size: usize,
    arg_idx: usize,
    label_no: usize,
    code: String,
}

impl FuncGenNasm {
    fn stack_alloc(&mut self, size: usize) -> usize {
        // FIXME: align allocation
        let offset = self.frame_size;
        self.frame_size += size;
        offset
    }

    fn next_ptr(&mut self) -> Val {
        Val::Loc(self.stack_alloc(Width::QWord as usize))
    }

    // Load the address of a value into a register
    fn gen_lea(&mut self, reg: Reg, val: &Val) {
        let dreg = reg_str(Width::QWord, reg);
        match val {
            Val::Void | Val::Imm(_) => unreachable!(),
            Val::Loc(offset) => writeln!(self.code, "lea {}, [rsp + {}]", dreg, offset).unwrap(),
            Val::Glo(name, offset) => writeln!(self.code, "lea {}, [{} + {}]", dreg, name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, reg, ptr);
                if *offset > 0 {
                    writeln!(self.code, "lea {}, [{} + {}]", dreg, dreg, offset).unwrap()
                }
            },
        }
    }

    // Load a value with a certain width to a register
    fn gen_load(&mut self, width: Width, reg: Reg, val: &Val) {
        let dreg = reg_str(width, reg);
        match val {
            Val::Void => unreachable!(),
            Val::Imm(val) => writeln!(self.code, "mov {}, {}", dreg, val).unwrap(),
            Val::Loc(offset) => writeln!(self.code, "mov {}, [rsp + {}]", dreg, offset).unwrap(),
            Val::Glo(name, offset) => writeln!(self.code, "mov {}, [{} + {}]", dreg, name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, reg, ptr);
                writeln!(self.code, "mov {}, [{} + {}]", dreg, reg_str(Width::QWord, reg), offset).unwrap()
            },
        }
    }

    // Store a register's contents into memory
    fn gen_store(&mut self, width: Width, val: &Val, reg: Reg, tmp_reg: Reg) {
        let sreg = reg_str(width, reg);
        match val {
            Val::Void | Val::Imm(_) => unreachable!(),
            Val::Loc(offset) => writeln!(self.code, "mov [rsp + {}], {}", offset, sreg).unwrap(),
            Val::Glo(name, offset) => writeln!(self.code, "mov [{} + {}], {}", name, offset, sreg).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, tmp_reg, ptr);
                writeln!(self.code, "mov [{} + {}], {}",
                    reg_str(Width::QWord, tmp_reg), offset, sreg).unwrap()
            },
        }
    }

    // Copy size bytes between two locations
    fn gen_copy(&mut self, mut dst: Val, mut src: Val, mut size: usize) {
        while size > 0 {
            // Find the maximum width we can copy
            let width = max_width(size);

            // Do the copy
            self.gen_load(width, Reg::Rax, &src);
            self.gen_store(width, &dst, Reg::Rax, Reg::Rbx);

            // Adjust for the next step
            size -= width as usize;
            if size > 0 {
                src = src.with_offset(width as usize);
                dst = dst.with_offset(width as usize);
            }
        }
    }

    // Load an arithmetic value
    fn gen_arith_load(&mut self, reg: Reg, ty: &Ty, val: &Val) -> &'static str {
        // Which instruction do we need, and what width do we extend to?
        let (insn, dreg, sloc) = match ty {
            // 8-bit/16-bit types extend to 32-bits
            Ty::U8 => ("movzx", reg_str(Width::DWord, reg), loc_str(Width::Byte)),
            Ty::I8 => ("movsx", reg_str(Width::DWord, reg), loc_str(Width::Byte)),
            Ty::U16 => ("movzx", reg_str(Width::DWord, reg), loc_str(Width::Word)),
            Ty::I16 => ("movsx", reg_str(Width::DWord, reg), loc_str(Width::Word)),
            // 32-bit and 64-bit types don't get extended
            Ty::U32|Ty::I32 => ("mov", reg_str(Width::DWord, reg), loc_str(Width::DWord)),
            Ty::U64|Ty::I64|Ty::USize => ("mov", reg_str(Width::QWord, reg), loc_str(Width::QWord)),
            _ => panic!("Expected arithmetic type"),
        };
        match val {
            Val::Void => unreachable!(),
            Val::Imm(val)
                => writeln!(self.code, "mov {}, {}", dreg, val).unwrap(),
            Val::Loc(offset)
                => writeln!(self.code, "{} {}, {} [rsp + {}]", insn, dreg, sloc, offset).unwrap(),
            Val::Glo(name, offset)
                => writeln!(self.code, "{} {}, {} [{} + {}]", insn, dreg, sloc, name, offset).unwrap(),
            Val::Deref(ptr, offset) => {
                self.gen_load(Width::QWord, reg, ptr);
                writeln!(self.code, "{} {}, {} [{} + {}]",
                    insn, dreg, sloc, reg_str(Width::QWord, reg), offset).unwrap()
            },
        }
        dreg
    }
}

impl FuncGen for FuncGenNasm {
    type Label = Label;
    type Val = Val;

    fn next_local(&mut self, ty: &Ty) -> Val {
        Val::Loc(self.stack_alloc(ty.get_size()))
    }

    fn next_label(&mut self) -> Label {
        let no = self.label_no;
        self.label_no += 1;
        Label(no)
    }

    fn tgt(&mut self, label: Label) {
        writeln!(self.code, ".{}:", label).unwrap();
    }

    fn jmp(&mut self, label: Label) {
        writeln!(self.code, "jmp .{}", label).unwrap();
    }

    fn arg_to_val(&mut self, ty: &Ty, dst: Val) {
        // Copy argument to destination
        // FIXME: this doesn't take into account non-register argument
        self.gen_store(type_width(ty), &dst, PARAMS[self.arg_idx], Reg::Rbx);
        // Increment argument index
        self.arg_idx += 1;
    }

    fn val_to_ret(&mut self, ty: &Ty, src: Val) {
        // FIXME: this doesn't take into account larger than 8 byte values
        self.gen_load(type_width(ty), Reg::Rax, &src);
    }

    fn bool_to_val(&mut self, ltrue: Label, lfalse: Label) -> Val {
        let off = self.stack_alloc(Ty::Bool.get_size());
        let lend = self.next_label();
        // True case
        writeln!(self.code, ".{}:", ltrue).unwrap();
        writeln!(self.code, "mov byte [rsp + {}], 1", off).unwrap();
        writeln!(self.code, "jmp .{}", lend).unwrap();
        // False case
        writeln!(self.code, ".{}:", lfalse).unwrap();
        writeln!(self.code, "mov byte [rsp + {}], 0", off).unwrap();
        writeln!(self.code, ".{}:", lend).unwrap();
        Val::Loc(off)
    }

    fn val_to_bool(&mut self, val: Val, ltrue: Label, lfalse: Label) {
        self.gen_load(Width::Byte, Reg::Rax, &val);
        writeln!(self.code, "test al, al").unwrap();
        writeln!(self.code, "jnz .{}\njmp .{}", ltrue, lfalse).unwrap();
    }

    fn immed(&mut self, val: usize) -> Val {
        Val::Imm(val)
    }

    fn global(&mut self, name: Rc<str>) -> Val {
        Val::Glo(name, 0)
    }

    fn compound(&mut self, ty: &Ty, vals: Vec<(&Ty, Val)>) -> Val {
        let off = self.stack_alloc(ty.get_size());
        let mut cur = off;
        for (ty, val) in vals.into_iter() {
            self.gen_copy(Val::Loc(cur), val, ty.get_size());
            cur += ty.get_size();
        }
        Val::Loc(off)
    }

    fn field(&mut self, val: Val, off: usize) -> Val {
        val.with_offset(off)
    }

    fn elem(&mut self, array: Val, index: Val, stride: usize) -> Val {
        if let Val::Imm(index) = index {
            // Avoid emitting multiply on constant index
            array.with_offset(index * stride)
        } else {
            // Generate a de-reference lvalue from a pointer to the element
            self.gen_lea(Reg::Rax, &array);
            self.gen_load(Width::QWord, Reg::Rbx, &index);
            writeln!(self.code, "imul rbx, {}\nadd rax, rbx", stride).unwrap();

            // Allocate temporary
            let tmp = self.next_ptr();
            self.gen_store(Width::QWord, &tmp, Reg::Rax, Reg::Rbx);
            Val::Deref(Box::new(tmp), 0)
        }
    }

    fn call(&mut self, retty: &Ty, varargs: bool, func: Val, args: Vec<(&Ty, Val)>) -> Val {
        // Move arguments to registers
        // FIXME: more than 6 arguments
        for (reg, (ty, val)) in PARAMS.iter().zip(args.iter()) {
            self.gen_load(type_width(ty), *reg, val);
        }

        // Move function address to rax
        self.gen_lea(Reg::Rbx, &func);

        // Generate call
        if varargs {
            writeln!(self.code, "xor eax, eax").unwrap();
        }
        writeln!(self.code, "call rbx").unwrap();

        if let Ty::Void = retty {
            // Create unusable value for precude
            Val::Void
        } else {
            // Otherwise move returned value to temporary
            let tmp = self.next_local(retty);
            self.gen_store(type_width(retty), &tmp, Reg::Rax, Reg::Rbx);
            tmp
        }
    }

    fn _ref(&mut self, val: Val) -> Val {
        // Load address to rax
        self.gen_lea(Reg::Rax, &val);
        // Save pointer to temporary
        let tmp = self.next_ptr();
        self.gen_store(Width::QWord, &tmp, Reg::Rax, Reg::Rbx);
        tmp
    }

    fn deref(&mut self, val: Val) -> Val {
        Val::Deref(Box::new(val), 0)
    }

    fn unary(&mut self, op: UOp, ty: &Ty, val: Val) -> Val {
        let reg = self.gen_arith_load(Reg::Rax, ty, &val);
        match op {
            UOp::Not => writeln!(self.code, "not {}", reg).unwrap(),
            UOp::Neg => writeln!(self.code, "neg {}", reg).unwrap(),
        }
        let tmp = self.next_local(ty);
        self.gen_store(type_width(ty), &tmp, Reg::Rax, Reg::Rbx);
        tmp
    }

    fn binary(&mut self, op: BOp, ty: &Ty, v1: Val, v2: Val) -> Val {
        // Load operands into registers
        let r1 = self.gen_arith_load(Reg::Rax, ty, &v1);
        let r2 = self.gen_arith_load(Reg::Rcx, ty, &v2);

        // Allocate temporary for the result
        let tmp = self.next_local(ty);

        match op {
            BOp::Mul => {
                // We only care about the lower half of the result, thus
                // we can use a two operand signed multiply everywhere
                writeln!(self.code, "imul {}, {}", r1, r2).unwrap()
            },
            BOp::Div | BOp::Rem => {
                // Clear upper half of dividend
                writeln!(self.code, "xor edx, edx").unwrap();
                // Choose instruction based on operand signedness
                if ty.is_signed() {
                    writeln!(self.code, "idiv {}", r2).unwrap();
                } else {
                    writeln!(self.code, "div {}", r2).unwrap();
                }
                // Result for remainder is handled differently
                if op == BOp::Rem {
                    self.gen_store(type_width(ty), &tmp, Reg::Rdx, Reg::Rbx);
                    return tmp;
                }
            },

            // These operations are sign independent with two's completement
            BOp::Add => writeln!(self.code, "add {}, {}", r1, r2).unwrap(),
            BOp::Sub => writeln!(self.code, "sub {}, {}", r1, r2).unwrap(),

            // The right operand can be any integer type, however it must have
            // a positive value less-than the number of bits in the left operand
            // otherwise this operation is undefined behavior
            BOp::Lsh => writeln!(self.code, "shl {}, cl", r1).unwrap(),
            BOp::Rsh => writeln!(self.code, "shr {}, cl", r1).unwrap(),

            // These operations are purely bitwise and ignore signedness
            BOp::And => writeln!(self.code, "and {}, {}", r1, r2).unwrap(),
            BOp::Xor => writeln!(self.code, "xor {}, {}", r1, r2).unwrap(),
            BOp::Or  => writeln!(self.code, "or {}, {}", r1, r2).unwrap(),
        }

        // Save result to temporary
        self.gen_store(type_width(ty), &tmp, Reg::Rax, Reg::Rbx);
        tmp
    }

    fn jcc(&mut self, cond: Cond, ltrue: Label, lfalse: Label, ty: &Ty, v1: Val, v2: Val) {
        let lhs_reg = self.gen_arith_load(Reg::Rax, ty, &v1);
        let rhs_reg = self.gen_arith_load(Reg::Rbx, ty, &v2);

        // Generate compare
        writeln!(self.code, "cmp {}, {}", lhs_reg, rhs_reg).unwrap();
        // Generate conditional jump to true case
        match cond {
            Cond::Lt => if ty.is_signed() {
                writeln!(self.code, "jl .{}", ltrue).unwrap();
            } else {
                writeln!(self.code, "jb .{}", ltrue).unwrap();
            },
            Cond::Le => if ty.is_signed() {
                writeln!(self.code, "jle .{}", ltrue).unwrap();
            } else {
                writeln!(self.code, "jbe .{}", ltrue).unwrap();
            },
            Cond::Gt => if ty.is_signed() {
                writeln!(self.code, "jg .{}", ltrue).unwrap();
            } else {
                writeln!(self.code, "ja .{}", ltrue).unwrap();
            },
            Cond::Ge => if ty.is_signed() {
                writeln!(self.code, "jge .{}", ltrue).unwrap();
            } else {
                writeln!(self.code, "jae .{}", ltrue).unwrap();
            },
            Cond::Eq => writeln!(self.code, "je .{}", ltrue).unwrap(),
            Cond::Ne => writeln!(self.code, "jne .{}", ltrue).unwrap(),
        }
        // Generate unconditional jump to false case
        writeln!(self.code, "jmp .{}", lfalse).unwrap();
    }

    fn assign(&mut self, ty: &Ty, dst: Val, src: Val) {
        self.gen_copy(dst, src, ty.get_size());
    }
}

pub struct GenNasm {
    // Linkage table
    str_no: usize,
    // String literal index
    linkage: Vec<(Rc<str>, Vis)>,
    // Sections
    text: String,
    rodata: String,
    data: String,
    bss: String,
}

impl GenNasm {
    pub fn new() -> GenNasm {
        GenNasm {
            str_no: 0,
            linkage: Vec::new(),
            text: String::new(),
            rodata: String::new(),
            data: String::new(),
            bss: String::new(),
        }
    }

    pub fn finalize<T: std::io::Write>(&self, output: &mut T) {
        // Write sections
        writeln!(output, "section .text\n{}", self.text).unwrap();
        writeln!(output, "section .rodata\n{}", self.rodata).unwrap();
        writeln!(output, "section .data\n{}", self.data).unwrap();
        writeln!(output, "section .bss\n{}", self.bss).unwrap();

        // Generate import/export table
        for (name, vis) in &self.linkage {
            match vis {
                Vis::Private => (),
                Vis::Export => writeln!(output, "global {}", name).unwrap(),
                Vis::Extern => writeln!(output, "extern {}", name).unwrap(),
            }
        }
    }
}

impl Gen for GenNasm {
    type DataGen = DataGenNasm;
    type FuncGen = FuncGenNasm;

    fn do_link(&mut self, name: Rc<str>, vis: Vis) {
        self.linkage.push((name, vis));
    }

    fn do_string(&mut self, chty: Ty, data: &str) -> (Rc<str>, Ty) {
        // Create assembly symbol
        let name: Rc<str> = format!("str${}", self.str_no).into();
        self.str_no += 1;
        // Generate data
        write!(self.rodata, "{} db ", name).unwrap();
        for byte in data.bytes() {
            write!(self.rodata, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(self.rodata, "0").unwrap();
        // Insert symbol
        let ty = Ty::Array {
            elem_type: Box::new(chty),
            elem_count: Some(data.len())
        };
        (name, ty)
    }

    fn do_bss(&mut self, name: &Rc<str>, ty: &Ty) {
        // Allocate bss entry
        // FIXME: align .bss entry
        writeln!(self.bss, "{} resb {}", name, ty.get_size()).unwrap();
    }

    fn begin_data(&mut self, name: &Rc<str>) -> DataGenNasm {
        // Generate heading
        writeln!(self.data, "{}:", name).unwrap();
        // Make data generator
        DataGenNasm {
            data: String::new()
        }
    }

    fn end_data(&mut self, data_gen: DataGenNasm) {
        // Merge generated data
        write!(self.data, "{}", data_gen.data).unwrap();
    }

    fn begin_func(&mut self, name: &Rc<str>) -> FuncGenNasm {
        // Create function heading
        writeln!(self.text, "{}:", name).unwrap();
        // Create generation context
        FuncGenNasm {
            frame_size: 0,
            arg_idx: 0,
            label_no: 0,
            code: String::new(),
        }
    }

    fn end_func(&mut self, func_gen: FuncGenNasm) {
        // Round stack frame
        let frame_size = (func_gen.frame_size + 15) / 16 * 16;
        // Write code
        writeln!(self.text,
            "push rbp\nmov rbp, rsp\nsub rsp, {}\n{}.$done:\nleave\nret",
            frame_size, func_gen.code).unwrap();
    }
}
