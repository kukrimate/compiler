# compiler-rs
Compiler for a simple programming language, written in Rust.

The grammar of the language is defined in (something resembling) EBNF
notation in `doc/grammar.txt`.

Current features:
- Strong typing, no implicit type conversions
- Signed and unsigned integer types of various widths
- Record (similar to C struct) types
- C like expressions
- Block based control flow
- Rust inspired sytax
- Assembly generation for x86\_64 (SysV ABI)

Notably missing features:
- Any "real" optimizations (we do constant folding)
- Pointer arithmetic (can be circumvented using casts)
- Struct return types and arguments (struct pointers work fine)
- Floating point math
- Many other bugs and TODOs

## license
GPLv2 only.
