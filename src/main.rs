// Base type for a pointer
enum PtrType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
}

// Bits of a pointer
type PtrBits = u64;

// Basic types with value
enum DataVal {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(u16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Ptr {
        base: PtrType,
        level: usize,
        value: PtrBits,
    }
}

enum Token {
    // Declarations
    Record,
    Union,
    Fn,
    Auto,
    Static,
    // Declaration markers
    Export,
    // Re-assingment (syntactically like a declaration)
    Set,
    // Constants
    Data(DataVal),
    // Indentifiers, labels
    Ident,
    Label,
    // Expression elements
    Add,    // +
    Sub,    // -
    Mul,    // *
    Div,    // /
    Rem,    // %
    Or,     // |
    And,    // &
    Xor,    // ^
    Lsh,    // <<
    Rsh,    // >>
    LParen, // (
    RParen, // )
    LSq,    // [
    RSq,    // ]
    LCurly, // {
    RCurly, // }
}

fn main() {

}
